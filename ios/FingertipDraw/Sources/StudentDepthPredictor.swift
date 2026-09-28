import CoreImage
import CoreImage.CIFilterBuiltins
import CoreML
import Foundation
import QuartzCore

final class StudentDepthPredictor {
    struct Prediction {
        let depthM: Float
        let latencyMS: Double
    }

    enum PredictorError: LocalizedError {
        case modelMissing(String)
        case modelMetadataMismatch(String)
        case modelInterfaceMismatch(String)
        case pixelBufferAllocation
        case resizeFailed
        case outputMissing
        case invalidOutput(Float)

        var errorDescription: String? {
            switch self {
            case .modelMissing(let name): return "\(name).mlmodelc がアプリに含まれていません"
            case .modelMetadataMismatch(let name):
                return "\(name) のCore ML metadataが期待する学習重み・変換方式と一致しません"
            case .modelInterfaceMismatch(let name):
                return "\(name) のCore ML入出力仕様が期待値と一致しません"
            case .pixelBufferAllocation: return "224×224入力バッファを作成できません"
            case .resizeFailed: return "カメラ画像を224×224へ変換できません"
            case .outputMissing: return "Core ML出力 depth_m がありません"
            case .invalidOutput(let value): return "無効な深さが出力されました: \(value)"
            }
        }
    }

    let descriptor: StudentDepthModelDescriptor
    private let model: MLModel
    private let context = CIContext(options: [.cacheIntermediates: false])
    private var pixelBufferPool: CVPixelBufferPool?

    init(
        descriptor: StudentDepthModelDescriptor,
        bundle: Bundle = .main
    ) throws {
        guard let url = bundle.url(
            forResource: descriptor.resourceName,
            withExtension: "mlmodelc"
        ) else {
            throw PredictorError.modelMissing(descriptor.resourceName)
        }
        let configuration = MLModelConfiguration()
        configuration.computeUnits = .all
        let loadedModel = try MLModel(contentsOf: url, configuration: configuration)
        try Self.validateModel(loadedModel, descriptor: descriptor)
        self.descriptor = descriptor
        model = loadedModel
        pixelBufferPool = Self.makePixelBufferPool()
    }

    func predict(
        sourcePixelBuffer: CVPixelBuffer,
        landmarksXY: [SIMD2<Float>]
    ) throws -> Prediction {
        precondition(landmarksXY.count == 4)
        let started = CACurrentMediaTime()
        let image = try resizedPixelBuffer(sourcePixelBuffer)
        let landmarks = try MLMultiArray(shape: [1, 4, 2], dataType: .float32)
        for (index, point) in landmarksXY.enumerated() {
            landmarks[index * 2] = NSNumber(value: point.x)
            landmarks[index * 2 + 1] = NSNumber(value: point.y)
        }
        let provider = try MLDictionaryFeatureProvider(dictionary: [
            "image": MLFeatureValue(pixelBuffer: image),
            "landmarks_xy": MLFeatureValue(multiArray: landmarks),
        ])
        let result = try model.prediction(from: provider)
        guard let array = result.featureValue(for: "depth_m")?.multiArrayValue else {
            throw PredictorError.outputMissing
        }
        let depthM = array[0].floatValue
        guard depthM.isFinite, depthM > 0 else {
            throw PredictorError.invalidOutput(depthM)
        }
        return Prediction(
            depthM: depthM,
            latencyMS: (CACurrentMediaTime() - started) * 1000.0
        )
    }

    private func resizedPixelBuffer(_ source: CVPixelBuffer) throws -> CVPixelBuffer {
        var destination: CVPixelBuffer?
        guard let pool = pixelBufferPool,
              CVPixelBufferPoolCreatePixelBuffer(nil, pool, &destination) == kCVReturnSuccess,
              let destination else {
            throw PredictorError.pixelBufferAllocation
        }
        let sourceWidth = Float(CVPixelBufferGetWidth(source))
        let sourceHeight = Float(CVPixelBufferGetHeight(source))
        guard sourceWidth > 0, sourceHeight > 0 else { throw PredictorError.resizeFailed }

        let filter = CIFilter.bicubicScaleTransform()
        filter.inputImage = CIImage(cvPixelBuffer: source)
        filter.scale = 224.0 / sourceHeight
        filter.aspectRatio = sourceHeight / sourceWidth
        // OpenCV INTER_CUBIC uses a = -0.75. B=0, C=0.75 is the matching
        // Keys-family cubic kernel available through Core Image.
        filter.parameterB = 0.0
        filter.parameterC = 0.75
        guard let image = filter.outputImage?.cropped(to: CGRect(x: 0, y: 0, width: 224, height: 224)) else {
            throw PredictorError.resizeFailed
        }
        context.render(
            image,
            to: destination,
            bounds: CGRect(x: 0, y: 0, width: 224, height: 224),
            colorSpace: CGColorSpaceCreateDeviceRGB()
        )
        return destination
    }

    private static func validateModel(
        _ model: MLModel,
        descriptor: StudentDepthModelDescriptor
    ) throws {
        let description = model.modelDescription
        guard let metadata = description.metadata[.creatorDefinedKey] as? [String: String],
              metadata["checkpoint_sha256"] == descriptor.checkpointSHA256,
              metadata["input_landmark_indices"] == "5,6,7,8",
              metadata["image_resize"] == "direct_bicubic_224x224_no_crop",
              metadata["target_device"] == "iPhone 15 (iPhone15,4), iOS 26.6.1" else {
            throw PredictorError.modelMetadataMismatch(descriptor.displayName)
        }
        if let expected = descriptor.runManifestSHA256,
           metadata["run_manifest_sha256"] != expected {
            throw PredictorError.modelMetadataMismatch(descriptor.displayName)
        }
        if let expected = descriptor.exportBackend,
           metadata["export_backend"] != expected {
            throw PredictorError.modelMetadataMismatch(descriptor.displayName)
        }
        if descriptor.runManifestSHA256 != nil,
           metadata["model_id"] != descriptor.id {
            throw PredictorError.modelMetadataMismatch(descriptor.displayName)
        }

        let inputs = description.inputDescriptionsByName
        guard Set(inputs.keys) == Set(["image", "landmarks_xy"]),
              let image = inputs["image"],
              image.type == .image,
              image.imageConstraint?.pixelsWide == 224,
              image.imageConstraint?.pixelsHigh == 224,
              let landmarks = inputs["landmarks_xy"],
              landmarks.type == .multiArray,
              landmarks.multiArrayConstraint?.shape.map({ $0.intValue }) == [1, 4, 2],
              landmarks.multiArrayConstraint?.dataType == .float32 else {
            throw PredictorError.modelInterfaceMismatch(descriptor.displayName)
        }

        let outputs = description.outputDescriptionsByName
        guard Set(outputs.keys) == Set(["depth_m"]),
              let depth = outputs["depth_m"],
              depth.type == .multiArray,
              depth.multiArrayConstraint?.shape.map({ $0.intValue }) == [1, 1],
              depth.multiArrayConstraint?.dataType == .float32 else {
            throw PredictorError.modelInterfaceMismatch(descriptor.displayName)
        }
    }

    private static func makePixelBufferPool() -> CVPixelBufferPool? {
        let attributes: [String: Any] = [
            kCVPixelBufferWidthKey as String: 224,
            kCVPixelBufferHeightKey as String: 224,
            kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA,
            kCVPixelBufferIOSurfacePropertiesKey as String: [:],
            kCVPixelBufferMetalCompatibilityKey as String: true,
        ]
        var pool: CVPixelBufferPool?
        CVPixelBufferPoolCreate(nil, nil, attributes as CFDictionary, &pool)
        return pool
    }
}
