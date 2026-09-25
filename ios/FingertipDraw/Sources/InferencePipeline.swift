import CoreMedia
import Foundation
import QuartzCore
import MediaPipeTasksVision
import UIKit

struct InferenceUpdate {
    let timestampMS: Int
    let landmarksXY: [SIMD2<Float>]
    let fingertipPixel: SIMD2<Float>
    let cameraPointM: SIMD3<Float>
    let intrinsics: CameraIntrinsics
    let handednessScore: Float?
    let handLatencyMS: Double
    let studentLatencyMS: Double
    let totalLatencyMS: Double
}

protocol InferencePipelineDelegate: AnyObject {
    func inferencePipeline(_ pipeline: InferencePipeline, didProduce update: InferenceUpdate)
    func inferencePipeline(_ pipeline: InferencePipeline, didFail message: String)
    func inferencePipeline(_ pipeline: InferencePipeline, didRejectCaptureConfiguration message: String)
}

final class InferencePipeline: NSObject {
    private struct Frame {
        let sampleBuffer: CMSampleBuffer
        let timestampMS: Int
        let submittedAt: CFTimeInterval
        let intrinsics: CameraIntrinsics
    }

    weak var delegate: InferencePipelineDelegate?

    private let queue = DispatchQueue(label: "jp.ac.fingertipdepth.inference")
    private let predictor: StudentDepthPredictor
    private var landmarker: HandLandmarker!
    private var inFlight: Frame?
    private var pendingLatest: Frame?
    private var lastTimestampMS = -1
    private var captureConditionFailureReported = false

    init(bundle: Bundle = .main) throws {
        predictor = try StudentDepthPredictor(bundle: bundle)
        super.init()
        guard let modelPath = bundle.path(forResource: "hand_landmarker", ofType: "task") else {
            throw PipelineError.handModelMissing
        }
        let options = HandLandmarkerOptions()
        options.baseOptions.modelAssetPath = modelPath
        options.runningMode = .liveStream
        options.numHands = 1
        options.minHandDetectionConfidence = 0.5
        options.minHandPresenceConfidence = 0.5
        options.minTrackingConfidence = 0.5
        options.handLandmarkerLiveStreamDelegate = self
        landmarker = try HandLandmarker(options: options)
    }

    func enqueue(_ sampleBuffer: CMSampleBuffer) {
        queue.async { [weak self] in
            guard let self else { return }
            let presentationTime = CMSampleBufferGetPresentationTimeStamp(sampleBuffer)
            let seconds = CMTimeGetSeconds(presentationTime)
            let candidate = seconds.isFinite ? Int((seconds * 1000.0).rounded()) : 0
            let timestampMS = max(candidate, self.lastTimestampMS + 1)
            self.lastTimestampMS = timestampMS
            let intrinsics = CameraIntrinsics.from(sampleBuffer: sampleBuffer)
            if let mismatch = FrameGeometry.captureConditionMismatch(for: intrinsics) {
                if !self.captureConditionFailureReported {
                    self.captureConditionFailureReported = true
                    DispatchQueue.main.async { [weak self] in
                        guard let self else { return }
                        self.delegate?.inferencePipeline(self, didRejectCaptureConfiguration: mismatch)
                    }
                }
                return
            }
            let frame = Frame(
                sampleBuffer: sampleBuffer,
                timestampMS: timestampMS,
                submittedAt: CACurrentMediaTime(),
                intrinsics: intrinsics
            )
            if self.inFlight == nil {
                self.submit(frame)
            } else {
                // There is never a growing queue: a newer camera frame replaces
                // the one that has not yet reached MediaPipe.
                self.pendingLatest = frame
            }
        }
    }

    private func submit(_ frame: Frame) {
        precondition(inFlight == nil)
        inFlight = frame
        do {
            let image = try MPImage(sampleBuffer: frame.sampleBuffer, orientation: .up)
            try landmarker.detectAsync(image: image, timestampInMilliseconds: frame.timestampMS)
        } catch {
            finishCurrentFrame(errorMessage: error.localizedDescription)
        }
    }

    private func consume(
        result: HandLandmarkerResult?,
        timestampMS: Int,
        error: Error?
    ) {
        guard let frame = inFlight, frame.timestampMS == timestampMS else {
            finishCurrentFrame(errorMessage: "MediaPipeの結果とカメラフレームを対応付けられません")
            return
        }
        if let error {
            finishCurrentFrame(errorMessage: error.localizedDescription)
            return
        }

        let handLatencyMS = (CACurrentMediaTime() - frame.submittedAt) * 1000.0
        guard let result, let hand = result.landmarks.first, hand.count > 8 else {
            finishCurrentFrame(errorMessage: nil)
            return
        }
        let indices = [5, 6, 7, 8]
        let landmarks = indices.map { SIMD2<Float>(hand[$0].x, hand[$0].y) }
        guard landmarks.allSatisfy({
            $0.x.isFinite && $0.y.isFinite
                && (0.0 ... 1.0).contains($0.x)
                && (0.0 ... 1.0).contains($0.y)
        }) else {
            finishCurrentFrame(errorMessage: nil)
            return
        }

        do {
            guard let pixelBuffer = CMSampleBufferGetImageBuffer(frame.sampleBuffer) else {
                throw PipelineError.pixelBufferMissing
            }
            let prediction = try predictor.predict(
                sourcePixelBuffer: pixelBuffer,
                landmarksXY: landmarks
            )
            let fingertipPixel = try FrameGeometry.pixel(
                from: landmarks[3],
                width: frame.intrinsics.width,
                height: frame.intrinsics.height
            )
            let cameraPoint = try FrameGeometry.backproject(
                pixel: fingertipPixel,
                depthM: prediction.depthM,
                intrinsics: frame.intrinsics
            )
            let handednessScore = result.handedness.first?.first?.score
            let update = InferenceUpdate(
                timestampMS: timestampMS,
                landmarksXY: landmarks,
                fingertipPixel: fingertipPixel,
                cameraPointM: cameraPoint,
                intrinsics: frame.intrinsics,
                handednessScore: handednessScore,
                handLatencyMS: handLatencyMS,
                studentLatencyMS: prediction.latencyMS,
                totalLatencyMS: (CACurrentMediaTime() - frame.submittedAt) * 1000.0
            )
            DispatchQueue.main.async { [weak self] in
                guard let self else { return }
                self.delegate?.inferencePipeline(self, didProduce: update)
            }
            finishCurrentFrame(errorMessage: nil)
        } catch {
            finishCurrentFrame(errorMessage: error.localizedDescription)
        }
    }

    private func finishCurrentFrame(errorMessage: String?) {
        inFlight = nil
        if let errorMessage {
            DispatchQueue.main.async { [weak self] in
                guard let self else { return }
                self.delegate?.inferencePipeline(self, didFail: errorMessage)
            }
        }
        if let next = pendingLatest {
            pendingLatest = nil
            submit(next)
        }
    }
}

extension InferencePipeline: HandLandmarkerLiveStreamDelegate {
    func handLandmarker(
        _ handLandmarker: HandLandmarker,
        didFinishDetection result: HandLandmarkerResult?,
        timestampInMilliseconds: Int,
        error: Error?
    ) {
        queue.async { [weak self] in
            self?.consume(
                result: result,
                timestampMS: timestampInMilliseconds,
                error: error
            )
        }
    }
}

enum PipelineError: LocalizedError {
    case handModelMissing
    case pixelBufferMissing

    var errorDescription: String? {
        switch self {
        case .handModelMissing: return "hand_landmarker.task がアプリに含まれていません"
        case .pixelBufferMissing: return "カメラフレームに画像バッファがありません"
        }
    }
}
