import Foundation

struct StudentDepthModelDescriptor: Equatable {
    let id: String
    let displayName: String
    let shortName: String
    let resourceName: String
    let checkpointSHA256: String
    let runManifestSHA256: String?
    let exportBackend: String?
}

enum StudentDepthModelVariant: String, CaseIterable, Hashable {
    case baseline
    case latest

    static let defaultVariant: StudentDepthModelVariant = .latest

    var descriptor: StudentDepthModelDescriptor {
        switch self {
        case .baseline:
            return StudentDepthModelDescriptor(
                id: "phase8_tail7",
                displayName: "旧モデル (Phase 8)",
                shortName: "旧",
                resourceName: "StudentDepth",
                checkpointSHA256: "e2e8941d2187e20dc716580fbbafb294cc9809db54b467a2d09fb52b39492252",
                runManifestSHA256: nil,
                exportBackend: nil
            )
        case .latest:
            return StudentDepthModelDescriptor(
                id: "latest_training_pipeline",
                displayName: "最新モデル (epoch 2)",
                shortName: "最新",
                resourceName: "StudentDepthLatest",
                checkpointSHA256: "66829e68ec5901d45c2805b62f4221b824bbbb2d215af7b14957c9efedbd600a",
                runManifestSHA256: "e699adf05b7f2f3507e3920304bc6b11ae135846df2f3fa8cb677879e33b2a2a",
                exportBackend: "torch.export"
            )
        }
    }
}
