// shape_2d_vertex.glsl
attribute vec2 Positions;
uniform mat4 ProjectionMatrix;
uniform mat4 ModelViewMatrix;
void main(void) {
    gl_Position = ProjectionMatrix * ModelViewMatrix * vec4(Positions, 0.0, 1.0);
}
