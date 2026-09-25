import XCTest
@testable import FingertipDraw

final class FrameGeometryTests: XCTestCase {
    func testPortraitPointIsConvertedBackToUnrotatedCaptureCoordinates() {
        XCTAssertEqual(
            FrameGeometry.captureDevicePoint(fromPortraitNormalized: SIMD2(0, 0)),
            SIMD2<Float>(0, 1)
        )
        XCTAssertEqual(
            FrameGeometry.captureDevicePoint(fromPortraitNormalized: SIMD2(1, 0)),
            SIMD2<Float>(0, 0)
        )
        XCTAssertEqual(
            FrameGeometry.captureDevicePoint(fromPortraitNormalized: SIMD2(0.25, 0.75)),
            SIMD2<Float>(0.75, 0.75)
        )
    }

    func testNormalizedPixelUsesHalfUpAndClampsBoundary() throws {
        XCTAssertEqual(
            try FrameGeometry.pixel(from: SIMD2(0.5, 0.5), width: 5, height: 3),
            SIMD2<Float>(3, 2)
        )
        XCTAssertEqual(
            try FrameGeometry.pixel(from: SIMD2(1.0, 1.0), width: 5, height: 3),
            SIMD2<Float>(4, 2)
        )
    }

    func testBackprojectionMatchesPinholeEquation() throws {
        let intrinsics = try CameraIntrinsics(
            fx: 1000,
            fy: 800,
            cx: 500,
            cy: 400,
            width: 1000,
            height: 800,
            source: .avFoundation
        )
        let point = try FrameGeometry.backproject(
            pixel: SIMD2(600, 320),
            depthM: 0.25,
            intrinsics: intrinsics
        )
        XCTAssertEqual(point.x, 0.025, accuracy: 1e-7)
        XCTAssertEqual(point.y, -0.025, accuracy: 1e-7)
        XCTAssertEqual(point.z, 0.25, accuracy: 1e-7)
    }

    func testApproximate36mmIntrinsicsAreCentred() {
        let intrinsics = CameraIntrinsics.approximate(width: 1080, height: 1920)
        XCTAssertEqual(intrinsics.cx, 539.5)
        XCTAssertEqual(intrinsics.cy, 959.5)
        XCTAssertEqual(intrinsics.fx, intrinsics.fy)
        XCTAssertGreaterThan(intrinsics.fx, 0)
        XCTAssertNil(FrameGeometry.captureConditionMismatch(for: intrinsics))
    }

    func testCaptureConditionRejectsWrongResolutionAndMeasuredFOV() throws {
        let wrongResolution = CameraIntrinsics.approximate(width: 720, height: 1280)
        XCTAssertNotNil(FrameGeometry.captureConditionMismatch(for: wrongResolution))

        let wrongFOV = try CameraIntrinsics(
            fx: 1300,
            fy: 1300,
            cx: 539.5,
            cy: 959.5,
            width: 1080,
            height: 1920,
            source: .avFoundation
        )
        XCTAssertNotNil(FrameGeometry.captureConditionMismatch(for: wrongFOV))
    }
}
