# graphics_matrix.py

from math import cos, sin, tan, sqrt
from graphics.graphics_float_bufferable import GraphicsFloatBufferable

class GraphicsMatrix(GraphicsFloatBufferable):

    def __init__(self) -> None:
        self.m = [
            1.0, 0.0, 0.0, 0.0,
            0.0, 1.0, 0.0, 0.0,
            0.0, 0.0, 1.0, 0.0,
            0.0, 0.0, 0.0, 1.0,
        ]

    # ---------------------------------------------------------
    # Basic / GraphicsFloatBufferable
    # ---------------------------------------------------------

    def reset(self) -> None:
        self.m = [
            1.0, 0.0, 0.0, 0.0,
            0.0, 1.0, 0.0, 0.0,
            0.0, 0.0, 1.0, 0.0,
            0.0, 0.0, 0.0, 1.0,
        ]

    def make(
        self,
        m00, m01, m02, m03,
        m10, m11, m12, m13,
        m20, m21, m22, m23,
        m30, m31, m32, m33,
    ) -> None:
        self.m[0]  = float(m00)
        self.m[1]  = float(m01)
        self.m[2]  = float(m02)
        self.m[3]  = float(m03)
        self.m[4]  = float(m10)
        self.m[5]  = float(m11)
        self.m[6]  = float(m12)
        self.m[7]  = float(m13)
        self.m[8]  = float(m20)
        self.m[9]  = float(m21)
        self.m[10] = float(m22)
        self.m[11] = float(m23)
        self.m[12] = float(m30)
        self.m[13] = float(m31)
        self.m[14] = float(m32)
        self.m[15] = float(m33)

    def copy(self) -> "GraphicsMatrix":
        m = GraphicsMatrix()
        m.m = self.m[:]
        return m

    def make_matrix(self, other: "GraphicsMatrix") -> None:
        self.m[:] = other.m[:]

    # GraphicsFloatBufferable
    def size(self) -> int:
        return 16

    def array(self):
        return list(self.m)

    def write_to_buffer(self, buffer) -> None:
        buffer.extend(self.m)

    def ortho(self, left: float, right: float,
              bottom: float, top: float,
              near_z: float, far_z: float) -> None:
        ral = right + left
        rsl = right - left
        tab = top + bottom
        tsb = top - bottom
        fan = far_z + near_z
        fsn = far_z - near_z

        self.make(
            2.0 / rsl, 0.0,        0.0,       0.0,
            0.0,       2.0 / tsb,  0.0,       0.0,
            0.0,       0.0,       -2.0 / fsn, 0.0,
            -ral / rsl, -tab / tsb, -fan / fsn, 1.0,
        )

    def ortho_size(self, width: float, height: float) -> None:
        self.ortho(
            left=0.0, right=width,
            bottom=height, top=0.0,
            near_z=-2048.0, far_z=0.0,
        )

    def translation(self, x: float, y: float, z: float) -> None:
        self.reset()
        self.m[12] = float(x)
        self.m[13] = float(y)
        self.m[14] = float(z)

    def translate(self, x: float, y: float, z: float) -> None:
        m = self.m
        tx = m[0] * x + m[4] * y + m[8]  * z + m[12]
        ty = m[1] * x + m[5] * y + m[9]  * z + m[13]
        tz = m[2] * x + m[6] * y + m[10] * z + m[14]
        tw = m[15]
        self.make(
            m[0], m[1], m[2], m[3],
            m[4], m[5], m[6], m[7],
            m[8], m[9], m[10], m[11],
            tx,   ty,   tz,   tw,
        )

    def scale(self, s: float) -> None:
        self.scale_xyz(s, s, s)

    def scale_xyz(self, sx: float, sy: float, sz: float) -> None:
        m = self.m
        self.make(
            m[0] * sx, m[1] * sx, m[2] * sx, m[3] * sx,
            m[4] * sy, m[5] * sy, m[6] * sy, m[7] * sy,
            m[8] * sz, m[9] * sz, m[10] * sz, m[11] * sz,
            m[12],    m[13],     m[14],      m[15],
        )

    def rotation_x(self, radians: float) -> None:
        c = cos(radians)
        s = sin(radians)
        self.make(
            1.0, 0.0, 0.0, 0.0,
            0.0,    c,    s, 0.0,
            0.0,   -s,    c, 0.0,
            0.0,  0.0,  0.0, 1.0
        )

    def rotate_x(self, radians: float) -> None:
        m = GraphicsMatrix()
        m.rotation_x(radians)
        self.multiply(m)

    def rotation_y(self, radians: float) -> None:
        c = cos(radians)
        s = sin(radians)
        self.make(
            c, 0.0,   -s, 0.0,
            0.0, 1.0, 0.0, 0.0,
            s, 0.0,    c, 0.0,
            0.0, 0.0, 0.0, 1.0
        )

    def rotate_y(self, radians: float) -> None:
        m = GraphicsMatrix()
        m.rotation_y(radians)
        self.multiply(m)

    def rotation_z(self, radians: float) -> None:
        c = cos(radians)
        s = sin(radians)
        self.make(
            c,   s, 0.0, 0.0,
            -s,   c, 0.0, 0.0,
            0.0, 0.0, 1.0, 0.0,
            0.0, 0.0, 0.0, 1.0
        )

    def rotate_z(self, radians: float) -> None:
        m = GraphicsMatrix()
        m.rotation_z(radians)
        self.multiply(m)

    def multiply(self, right: "GraphicsMatrix") -> None:
        a0, a1, a2, a3,  a4, a5, a6, a7,  a8, a9, a10, a11,  a12, a13, a14, a15 = self.m
        b0, b1, b2, b3,  b4, b5, b6, b7,  b8, b9, b10, b11,  b12, b13, b14, b15 = right.m
        r0  = a0 * b0  + a4 * b1  + a8 * b2   + a12 * b3
        r4  = a0 * b4  + a4 * b5  + a8 * b6   + a12 * b7
        r8  = a0 * b8  + a4 * b9  + a8 * b10  + a12 * b11
        r12 = a0 * b12 + a4 * b13 + a8 * b14  + a12 * b15
        r1  = a1 * b0  + a5 * b1  + a9 * b2   + a13 * b3
        r5  = a1 * b4  + a5 * b5  + a9 * b6   + a13 * b7
        r9  = a1 * b8  + a5 * b9  + a9 * b10  + a13 * b11
        r13 = a1 * b12 + a5 * b13 + a9 * b14  + a13 * b15
        r2  = a2 * b0  + a6 * b1  + a10 * b2  + a14 * b3
        r6  = a2 * b4  + a6 * b5  + a10 * b6  + a14 * b7
        r10 = a2 * b8  + a6 * b9  + a10 * b10 + a14 * b11
        r14 = a2 * b12 + a6 * b13 + a10 * b14 + a14 * b15
        r3  = a3 * b0  + a7 * b1  + a11 * b2  + a15 * b3
        r7  = a3 * b4  + a7 * b5  + a11 * b6  + a15 * b7
        r11 = a3 * b8  + a7 * b9  + a11 * b10 + a15 * b11
        r15 = a3 * b12 + a7 * b13 + a11 * b14 + a15 * b15
        self.m = [
            r0, r1, r2, r3,
            r4, r5, r6, r7,
            r8, r9, r10, r11,
            r12, r13, r14, r15,
        ]

    def add(self, right: "GraphicsMatrix") -> None:
        m = [a + b for a, b in zip(self.m, right.m)]
        self.m = m

    def subtract(self, right: "GraphicsMatrix") -> None:
        m = [a - b for a, b in zip(self.m, right.m)]
        self.m = m

    def determinant(self) -> float:
        m = self.m
        return (
            m[3] * m[6] * m[9] * m[12] - m[2] * m[7] * m[9] * m[12]
            - m[3] * m[5] * m[10] * m[12] + m[1] * m[7] * m[10] * m[12]
            + m[2] * m[5] * m[11] * m[12] - m[1] * m[6] * m[11] * m[12]
            - m[3] * m[6] * m[8] * m[13] + m[2] * m[7] * m[8] * m[13]
            + m[3] * m[4] * m[10] * m[13] - m[0] * m[7] * m[10] * m[13]
            - m[2] * m[4] * m[11] * m[13] + m[0] * m[6] * m[11] * m[13]
            + m[3] * m[5] * m[8] * m[14] - m[1] * m[7] * m[8] * m[14]
            - m[3] * m[4] * m[9] * m[14] + m[0] * m[7] * m[9] * m[14]
            + m[1] * m[4] * m[11] * m[14] - m[0] * m[5] * m[11] * m[14]
            - m[2] * m[5] * m[8] * m[15] + m[1] * m[6] * m[8] * m[15]
            + m[2] * m[4] * m[9] * m[15] - m[0] * m[6] * m[9] * m[15]
            - m[1] * m[4] * m[10] * m[15] + m[0] * m[5] * m[10] * m[15]
        )

    def invert(self) -> None:
        m = self.m[:]
        inv = [0.0] * 16

        inv[0] = m[6] * m[11] * m[13] - m[7] * m[10] * m[13] + m[7] * m[9] * m[14] - m[5] * m[11] * m[14] - m[6] * m[9] * m[15] + m[5] * m[10] * m[15]
        inv[1] = m[3] * m[10] * m[13] - m[2] * m[11] * m[13] - m[3] * m[9] * m[14] + m[1] * m[11] * m[14] + m[2] * m[9] * m[15] - m[1] * m[10] * m[15]
        inv[2] = m[2] * m[7] * m[13] - m[3] * m[6] * m[13] + m[3] * m[5] * m[14] - m[1] * m[7] * m[14] - m[2] * m[5] * m[15] + m[1] * m[6] * m[15]
        inv[3] = m[3] * m[6] * m[9] - m[2] * m[7] * m[9] - m[3] * m[5] * m[10] + m[1] * m[7] * m[10] + m[2] * m[5] * m[11] - m[1] * m[6] * m[11]
        inv[4] = m[7] * m[10] * m[12] - m[6] * m[11] * m[12] - m[7] * m[8] * m[14] + m[4] * m[11] * m[14] + m[6] * m[8] * m[15] - m[4] * m[10] * m[15]
        inv[5] = m[2] * m[11] * m[12] - m[3] * m[10] * m[12] + m[3] * m[8] * m[14] - m[0] * m[11] * m[14] - m[2] * m[8] * m[15] + m[0] * m[10] * m[15]
        inv[6] = m[3] * m[6] * m[12] - m[2] * m[7] * m[12] - m[3] * m[4] * m[14] + m[0] * m[7] * m[14] + m[2] * m[4] * m[15] - m[0] * m[6] * m[15]
        inv[7] = m[2] * m[7] * m[8] - m[3] * m[6] * m[8] + m[3] * m[4] * m[10] - m[0] * m[7] * m[10] - m[2] * m[4] * m[11] + m[0] * m[6] * m[11]
        inv[8] = m[5] * m[11] * m[12] - m[7] * m[9] * m[12] + m[7] * m[8] * m[13] - m[4] * m[11] * m[13] - m[5] * m[8] * m[15] + m[4] * m[9] * m[15]
        inv[9] = m[3] * m[9] * m[12] - m[1] * m[11] * m[12] - m[3] * m[8] * m[13] + m[0] * m[11] * m[13] + m[1] * m[8] * m[15] - m[0] * m[9] * m[15]
        inv[10] = m[1] * m[7] * m[12] - m[3] * m[5] * m[12] + m[3] * m[4] * m[13] - m[0] * m[7] * m[13] - m[1] * m[4] * m[15] + m[0] * m[5] * m[15]
        inv[11] = m[3] * m[5] * m[8] - m[1] * m[7] * m[8] - m[3] * m[4] * m[9] + m[0] * m[7] * m[9] + m[1] * m[4] * m[11] - m[0] * m[5] * m[11]
        inv[12] = m[6] * m[9] * m[12] - m[5] * m[10] * m[12] - m[6] * m[8] * m[13] + m[4] * m[10] * m[13] + m[5] * m[8] * m[14] - m[4] * m[9] * m[14]
        inv[13] = m[1] * m[10] * m[12] - m[2] * m[9] * m[12] + m[2] * m[8] * m[13] - m[0] * m[10] * m[13] - m[1] * m[8] * m[14] + m[0] * m[9] * m[14]
        inv[14] = m[2] * m[5] * m[12] - m[1] * m[6] * m[12] - m[2] * m[4] * m[13] + m[0] * m[6] * m[13] + m[1] * m[4] * m[14] - m[0] * m[5] * m[14]
        inv[15] = m[1] * m[6] * m[8] - m[2] * m[5] * m[8] + m[2] * m[4] * m[9] - m[0] * m[6] * m[9] - m[1] * m[4] * m[10] + m[0] * m[5] * m[10]

        det = (
            m[3] * m[6] * m[9] * m[12] - m[2] * m[7] * m[9] * m[12]
            - m[3] * m[5] * m[10] * m[12] + m[1] * m[7] * m[10] * m[12]
            + m[2] * m[5] * m[11] * m[12] - m[1] * m[6] * m[11] * m[12]
            - m[3] * m[6] * m[8] * m[13] + m[2] * m[7] * m[8] * m[13]
            + m[3] * m[4] * m[10] * m[13] - m[0] * m[7] * m[10] * m[13]
            - m[2] * m[4] * m[11] * m[13] + m[0] * m[6] * m[11] * m[13]
            + m[3] * m[5] * m[8] * m[14] - m[1] * m[7] * m[8] * m[14]
            - m[3] * m[4] * m[9] * m[14] + m[0] * m[7] * m[9] * m[14]
            + m[1] * m[4] * m[11] * m[14] - m[0] * m[5] * m[11] * m[14]
            - m[2] * m[5] * m[8] * m[15] + m[1] * m[6] * m[8] * m[15]
            + m[2] * m[4] * m[9] * m[15] - m[0] * m[6] * m[9] * m[15]
            - m[1] * m[4] * m[10] * m[15] + m[0] * m[5] * m[10] * m[15]
        )

        if det != 0.0:
            inv_det = 1.0 / det
            self.m = [v * inv_det for v in inv]
        else:
            # Non-invertible; leave as-is or reset
            pass
