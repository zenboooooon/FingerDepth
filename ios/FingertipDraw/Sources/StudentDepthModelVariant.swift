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
                displayName: "最新モデル (80 cm・epoch 13)",
                shortName: "最新",
                resourceName: "StudentDepthLatest",
                checkpointSHA256: "b74ffbe49fd98f96a4d3280074fef148df27ac1016d1fcfeca5cc905fd4f84e5",
                runManifestSHA256: "80797bb0529184bbbd4d0f8407772c45e9733098112aa07ec933f648db615e19",
                exportBackend: "torch.export"
            )
        }
    }
}
