import CoreMedia
import CoreVideo
import Foundation
import simd

struct CameraIntrinsics: Equatable {
    enum Source: String {
        case avFoundation
        case approximate36mmEquivalent
    }

    let fx: Float
    let fy: Float
    let cx: Float
    let cy: Float
    let width: Int
    let height: Int
    let source: Source

    init(
        fx: Float,
        fy: Float,
        cx: Float,
        cy: Float,
        width: Int,
        height: Int,
        source: Source
    ) throws {
        guard width > 0, height > 0,
              fx.isFinite, fy.isFinite, cx.isFinite, cy.isFinite,
              fx > 0, fy > 0 else {
            throw GeometryError.invalidIntrinsics
        }
        self.fx = fx
        self.fy = fy
        self.cx = cx
        self.cy = cy
        self.width = width
        self.height = height
        self.source = source
    }

    static func from(sampleBuffer: CMSampleBuffer) -> CameraIntrinsics {
        guard let imageBuffer = CMSampleBufferGetImageBuffer(sampleBuffer) else {
            return approximate(width: TargetConfiguration.captureWidth, height: TargetConfiguration.captureHeight)
        }
        let width = CVPixelBufferGetWidth(imageBuffer)
        let height = CVPixelBufferGetHeight(imageBuffer)
        guard
            let attachment = CMGetAttachment(
                sampleBuffer,
                key: kCMSampleBufferAttachmentKey_CameraIntrinsicMatrix,
                attachmentModeOut: nil
            ) as? Data,
            attachment.count == MemoryLayout<matrix_float3x3>.size
        else {
            return approximate(width: width, height: height)
        }

        var matrix = matrix_float3x3()
        _ = withUnsafeMutableBytes(of: &matrix) { destination in
            attachment.copyBytes(to: destination)
        }
        let raw = (
            fx: matrix.columns.0.x,
            fy: matrix.columns.1.y,
            cx: matrix.columns.2.x,
            cy: matrix.columns.2.y
        )

        // Some capture paths keep K in the unrotated 1920x1080 coordinate
        // system even when the delivered CVPixelBuffer is portrait. Compare
        // both principal points and retain the interpretation closest to the
        // actual buffer centre.
        let centerX = Float(width - 1) / 2.0
        let centerY = Float(height - 1) / 2.0
        let rawError = hypotf(raw.cx - centerX, raw.cy - centerY)
        let rotated = (
            fx: raw.fy,
            fy: raw.fx,
            cx: Float(width - 1) - raw.cy,
            cy: raw.cx
        )
        let rotatedError = hypotf(rotated.cx - centerX, rotated.cy - centerY)
        let selected = rotatedError + 1.0 < rawError ? rotated : raw
        return (try? CameraIntrinsics(
            fx: selected.fx,
            fy: selected.fy,
            cx: selected.cx,
            cy: selected.cy,
            width: width,
            height: height,
            source: .avFoundation
        )) ?? approximate(width: width, height: height)
    }

    static func approximate(width: Int, height: Int) -> CameraIntrinsics {
        let diagonalPixels = hypotf(Float(width), Float(height))
        let fullFrameDiagonalMM: Float = 43.266615
        let focalPixels = TargetConfiguration.requestedEquivalentFocalLengthMM
            / fullFrameDiagonalMM * diagonalPixels
        return try! CameraIntrinsics(
            fx: focalPixels,
            fy: focalPixels,
            cx: Float(width - 1) / 2.0,
            cy: Float(height - 1) / 2.0,
            width: width,
            height: height,
            source: .approximate36mmEquivalent
        )
    }
}

enum GeometryError: Error {
    case invalidIntrinsics
    case invalidNormalizedPoint
    case invalidDepth
}

enum FrameGeometry {
    /// AVCaptureVideoPreviewLayer expects normalized coordinates in the
    /// unrotated landscape capture-device coordinate system. MediaPipe sees
    /// the physically rotated portrait CVPixelBuffer produced by the 90°
    /// clockwise output connection, so undo that rotation before conversion.
    static func captureDevicePoint(
        fromPortraitNormalized point: SIMD2<Float>
    ) -> SIMD2<Float> {
        SIMD2(point.y, 1.0 - point.x)
    }

    static func captureConditionMismatch(
        for intrinsics: CameraIntrinsics
    ) -> String? {
        guard intrinsics.width == TargetConfiguration.captureWidth,
              intrinsics.height == TargetConfiguration.captureHeight else {
            return "カメラ解像度が学習条件と異なります: \(intrinsics.width)×\(intrinsics.height)（必要: \(TargetConfiguration.captureWidth)×\(TargetConfiguration.captureHeight)）"
        }

        // The fallback K is the same 36 mm-equivalent approximation used to
        // create the training labels. When AVFoundation supplies measured K,
        // reject a materially different FOV instead of silently changing the
        // RGB-to-depth scale learned by the student.
        guard intrinsics.source == .avFoundation else { return nil }
        let expected = TargetConfiguration.trainingFocalLengthPixels
        let largestRelativeError = max(
            abs(intrinsics.fx - expected) / expected,
            abs(intrinsics.fy - expected) / expected
        )
        guard largestRelativeError <= TargetConfiguration.focalLengthRelativeTolerance else {
            return String(
                format: "カメラFOVが学習条件と異なります: fx=%.1f, fy=%.1f px（基準 %.1f px、許容 ±%.0f%%）",
                intrinsics.fx,
                intrinsics.fy,
                expected,
                TargetConfiguration.focalLengthRelativeTolerance * 100.0
            )
        }
        return nil
    }

    static func pixel(
        from normalizedPoint: SIMD2<Float>,
        width: Int,
        height: Int
    ) throws -> SIMD2<Float> {
        guard width > 0, height > 0,
              normalizedPoint.x.isFinite, normalizedPoint.y.isFinite,
              (0.0 ... 1.0).contains(normalizedPoint.x),
              (0.0 ... 1.0).contains(normalizedPoint.y) else {
            throw GeometryError.invalidNormalizedPoint
        }
        let u = floorf(normalizedPoint.x * Float(width) + 0.5)
        let v = floorf(normalizedPoint.y * Float(height) + 0.5)
        return SIMD2(
            min(max(u, 0), Float(width - 1)),
            min(max(v, 0), Float(height - 1))
        )
    }

    static func backproject(
        pixel: SIMD2<Float>,
        depthM: Float,
        intrinsics: CameraIntrinsics
    ) throws -> SIMD3<Float> {
        guard depthM.isFinite, depthM > 0 else { throw GeometryError.invalidDepth }
        return SIMD3(
            (pixel.x - intrinsics.cx) * depthM / intrinsics.fx,
            (pixel.y - intrinsics.cy) * depthM / intrinsics.fy,
            depthM
        )
    }
}
