import XCTest
@testable import FingertipDraw

final class StudentDepthModelVariantTests: XCTestCase {
    func testCatalogHasUniqueStableIdentifiersAndResources() {
        let variants = StudentDepthModelVariant.allCases
        let descriptors = variants.map(\.descriptor)

        XCTAssertEqual(StudentDepthModelVariant.defaultVariant, .latest)
        XCTAssertEqual(Set(descriptors.map(\.id)).count, descriptors.count)
        XCTAssertEqual(Set(descriptors.map(\.resourceName)).count, descriptors.count)
        XCTAssertEqual(
            Set(descriptors.map(\.checkpointSHA256)).count,
            descriptors.count
        )
    }

    func testEveryCheckpointAndOptionalRunDigestIsSHA256() {
        for descriptor in StudentDepthModelVariant.allCases.map(\.descriptor) {
            XCTAssertTrue(isSHA256(descriptor.checkpointSHA256))
            if let runManifestSHA256 = descriptor.runManifestSHA256 {
                XCTAssertTrue(isSHA256(runManifestSHA256))
            }
        }
    }

    func testLatestModelUsesTorchExportArtifact() {
        let latest = StudentDepthModelVariant.latest.descriptor

        XCTAssertEqual(latest.resourceName, "StudentDepthLatest")
        XCTAssertEqual(latest.exportBackend, "torch.export")
        XCTAssertNotNil(latest.runManifestSHA256)
    }

    func testEveryBundledModelMatchesItsDescriptorAndInterface() throws {
        for descriptor in StudentDepthModelVariant.allCases.map(\.descriptor) {
            _ = try StudentDepthPredictor(descriptor: descriptor, bundle: .main)
        }
    }

    private func isSHA256(_ value: String) -> Bool {
        value.count == 64 && value.allSatisfy { $0.isHexDigit && !$0.isUppercase }
    }
}
