import AVFoundation
import UIKit

struct OverlaySnapshot {
    let selectedLandmarks: [SIMD2<Float>]
    let currentPointM: SIMD3<Float>
    let trajectory: [TrajectorySample]
}

final class OverlayView: UIView {
    weak var previewLayer: AVCaptureVideoPreviewLayer?
    var snapshot: OverlaySnapshot? {
        didSet { setNeedsDisplay() }
    }

    override init(frame: CGRect) {
        super.init(frame: frame)
        backgroundColor = .clear
        isUserInteractionEnabled = false
        contentMode = .redraw
    }

    required init?(coder: NSCoder) {
        fatalError("init(coder:) has not been implemented")
    }

    override func draw(_ rect: CGRect) {
        guard let context = UIGraphicsGetCurrentContext(),
              let snapshot,
              let previewLayer else { return }
        drawCameraTrajectory(snapshot, previewLayer: previewLayer, context: context)
        drawLandmarks(snapshot, previewLayer: previewLayer, context: context)
        drawXZPanel(snapshot, context: context)
    }

    private func viewPoint(
        _ normalized: SIMD2<Float>,
        previewLayer: AVCaptureVideoPreviewLayer
    ) -> CGPoint {
        let capturePoint = FrameGeometry.captureDevicePoint(
            fromPortraitNormalized: normalized
        )
        return previewLayer.layerPointConverted(
            fromCaptureDevicePoint: CGPoint(
                x: CGFloat(capturePoint.x),
                y: CGFloat(capturePoint.y)
            )
        )
    }

    private func drawCameraTrajectory(
        _ snapshot: OverlaySnapshot,
        previewLayer: AVCaptureVideoPreviewLayer,
        context: CGContext
    ) {
        guard !snapshot.trajectory.isEmpty else { return }
        let path = UIBezierPath()
        for (index, sample) in snapshot.trajectory.enumerated() {
            let point = viewPoint(sample.fingertipNormalized, previewLayer: previewLayer)
            index == 0 ? path.move(to: point) : path.addLine(to: point)
        }
        context.saveGState()
        context.setShadow(offset: .zero, blur: 4, color: UIColor.black.withAlphaComponent(0.8).cgColor)
        UIColor.systemCyan.setStroke()
        path.lineWidth = 4
        path.lineJoinStyle = .round
        path.lineCapStyle = .round
        path.stroke()
        context.restoreGState()
    }

    private func drawLandmarks(
        _ snapshot: OverlaySnapshot,
        previewLayer: AVCaptureVideoPreviewLayer,
        context: CGContext
    ) {
        guard snapshot.selectedLandmarks.count == 4 else { return }
        let points = snapshot.selectedLandmarks.map { viewPoint($0, previewLayer: previewLayer) }
        let bone = UIBezierPath()
        bone.move(to: points[0])
        points.dropFirst().forEach { bone.addLine(to: $0) }
        UIColor.systemOrange.setStroke()
        bone.lineWidth = 3
        bone.stroke()
        for (index, point) in points.enumerated() {
            let radius: CGFloat = index == 3 ? 8 : 5
            let circle = UIBezierPath(
                ovalIn: CGRect(
                    x: point.x - radius,
                    y: point.y - radius,
                    width: radius * 2,
                    height: radius * 2
                )
            )
            (index == 3 ? UIColor.systemCyan : UIColor.systemOrange).setFill()
            circle.fill()
        }
    }

    private func drawXZPanel(_ snapshot: OverlaySnapshot, context: CGContext) {
        let panelWidth = min(bounds.width * 0.46, 190)
        let panel = CGRect(x: bounds.maxX - panelWidth - 12, y: 92, width: panelWidth, height: 170)
        let background = UIBezierPath(roundedRect: panel, cornerRadius: 12)
        UIColor.black.withAlphaComponent(0.70).setFill()
        background.fill()

        let plot = panel.insetBy(dx: 14, dy: 27)
        let points = snapshot.trajectory.map(\.cameraPointM) + [snapshot.currentPointM]
        guard let minX = points.map(\.x).min(), let maxX = points.map(\.x).max(),
              let minZ = points.map(\.z).min(), let maxZ = points.map(\.z).max() else { return }
        let xPadding = max((maxX - minX) * 0.10, 0.01)
        let zPadding = max((maxZ - minZ) * 0.10, 0.02)
        let xLow = minX - xPadding
        let xSpan = max(maxX - minX + 2 * xPadding, 0.02)
        let zLow = minZ - zPadding
        let zSpan = max(maxZ - minZ + 2 * zPadding, 0.04)
        func map(_ value: SIMD3<Float>) -> CGPoint {
            CGPoint(
                x: plot.minX + CGFloat((value.x - xLow) / xSpan) * plot.width,
                y: plot.minY + CGFloat((value.z - zLow) / zSpan) * plot.height
            )
        }

        if !snapshot.trajectory.isEmpty {
            let path = UIBezierPath()
            for (index, sample) in snapshot.trajectory.enumerated() {
                let point = map(sample.cameraPointM)
                index == 0 ? path.move(to: point) : path.addLine(to: point)
            }
            UIColor.systemCyan.setStroke()
            path.lineWidth = 2
            path.stroke()
        }
        let current = map(snapshot.currentPointM)
        UIColor.white.setFill()
        UIBezierPath(ovalIn: CGRect(x: current.x - 4, y: current.y - 4, width: 8, height: 8)).fill()

        let title = "X–Z trajectory"
        title.draw(
            at: CGPoint(x: panel.minX + 12, y: panel.minY + 7),
            withAttributes: [
                .font: UIFont.monospacedSystemFont(ofSize: 12, weight: .semibold),
                .foregroundColor: UIColor.white,
            ]
        )
        let range = String(format: "Z %.1f–%.1f cm", zLow * 100, (zLow + zSpan) * 100)
        range.draw(
            at: CGPoint(x: panel.minX + 12, y: panel.maxY - 21),
            withAttributes: [
                .font: UIFont.monospacedSystemFont(ofSize: 10, weight: .regular),
                .foregroundColor: UIColor.white.withAlphaComponent(0.8),
            ]
        )
    }
}
