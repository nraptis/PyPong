// sprite_2d_fragment.glsl
uniform vec4 ModulateColor;
varying vec2 TextureCoordinatesOut;
uniform sampler2D Texture;
void main(void) {
    gl_FragColor = ModulateColor * texture2D(Texture, TextureCoordinatesOut);
}
