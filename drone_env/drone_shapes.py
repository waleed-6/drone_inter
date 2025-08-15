import pybullet as p

def create_drone_visual(
    body_radius=0.25,
    body_height=0.08,
    arm_half=0.80,
    arm_thickness=0.06,
    rotor_radius=0.18,
    rotor_thickness=0.02,
    color_body=(0.1, 0.5, 1.0, 1.0),
    color_arm=(0.15, 0.15, 0.15, 1.0),
    color_rotor=(0.05, 0.05, 0.05, 1.0),
):
    """
    Returns a visual shape index for a compound, drone-like shape:
      - center cylinder 'can' (body)
      - two cross arms (boxes) on X and Y
      - four thin rotor discs at arm tips
    Orientation: Z is up. All offsets are in base frame.
    """
    # Shapes
    shapeTypes = []
    halfExtents = []
    radii = []
    lengths = []
    rgbaColors = []
    visPos = []
    visOri = []

    # Helpers
    def add_box(halfX, halfY, halfZ, color, pos, euler=(0, 0, 0)):
        shapeTypes.append(p.GEOM_BOX)
        halfExtents.append([halfX, halfY, halfZ])
        radii.append(0)      # unused
        lengths.append(0)    # unused
        rgbaColors.append(color)
        visPos.append(pos)
        visOri.append(p.getQuaternionFromEuler(euler))

    def add_cyl(radius, length, color, pos, euler=(0, 0, 0)):
        shapeTypes.append(p.GEOM_CYLINDER)
        halfExtents.append([0, 0, 0])  # unused
        radii.append(radius)
        lengths.append(length)
        rgbaColors.append(color)
        visPos.append(pos)
        visOri.append(p.getQuaternionFromEuler(euler))

    # 1) Central body (upright cylinder)
    add_cyl(body_radius, body_height, color_body, [0, 0, body_height * 0.5])

    # 2) Cross arms
    # X-arm (long along X)
    add_box(arm_half, arm_thickness * 0.5, arm_thickness * 0.5, color_arm,
            [0, 0, body_height * 0.5])
    # Y-arm (long along Y)
    add_box(arm_thickness * 0.5, arm_half, arm_thickness * 0.5, color_arm,
            [0, 0, body_height * 0.5])

    # 3) Rotors (flat cylinders) at four ends, slightly above body top
    zr = body_height + rotor_thickness * 0.5
    tip = arm_half + rotor_radius * 0.1  # tiny gap so discs don't intersect arms
    add_cyl(rotor_radius, rotor_thickness, color_rotor, [ tip,  0.0, zr])
    add_cyl(rotor_radius, rotor_thickness, color_rotor, [-tip,  0.0, zr])
    add_cyl(rotor_radius, rotor_thickness, color_rotor, [ 0.0,  tip, zr])
    add_cyl(rotor_radius, rotor_thickness, color_rotor, [ 0.0, -tip, zr])

    return p.createVisualShapeArray(
        shapeTypes=shapeTypes,
        halfExtents=halfExtents,
        radii=radii,
        lengths=lengths,
        rgbaColors=rgbaColors,
        visualFramePositions=visPos,
        visualFrameOrientations=visOri,
    )
