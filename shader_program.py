from OpenGL.GL import (
    glCreateProgram,
    glAttachShader,
    glLinkProgram,
    glGetProgramiv,
    glGetProgramInfoLog,
    glDeleteProgram,
    glGetAttribLocation,
    glGetUniformLocation,
    GL_LINK_STATUS,
)

class ShaderProgram:
    
    def __init__(self, name: str, vertex_shader: int, fragment_shader: int):
        self.name = name
        self.vertex_shader = vertex_shader
        self.fragment_shader = fragment_shader

        self.program: int = 0

        # Attribute locations
        self.attribute_location_position = -1
        self.attribute_location_texture_coordinates = -1

        # Uniform locations
        self.uniform_location_texture = -1
        self.uniform_location_modulate_color = -1
        self.uniform_location_projection_matrix = -1
        self.uniform_location_model_view_matrix = -1
        self.uniform_location_texture_size = -1

        # Attribute layout info (you can fill these in per subclass)
        self.attribute_stride_position = -1
        self.attribute_size_position = -1
        self.attribute_offset_position = -1

        self.attribute_stride_texture_coordinates = -1
        self.attribute_size_texture_coordinates = -1
        self.attribute_offset_texture_coordinates = -1

        # Create and link program
        if (vertex_shader > 0) and (fragment_shader > 0):
            self.program = self._load_program(vertex_shader, fragment_shader)
            print(
                f"==> Success! Created Shader Program [{name}], "
                f"vertexShader: {vertex_shader}, fragmentShader: {fragment_shader}, "
                f"program = {self.program}"
            )

            # Check link status
            link_status = glGetProgramiv(self.program, GL_LINK_STATUS)
            if link_status == 0:
                log = glGetProgramInfoLog(self.program)
                print(f"[ShaderLink] Error linking program {name}: {log}")
                glDeleteProgram(self.program)
                self.program = 0
        else:
            print(
                f"==> Failed! Created Shader Program [{name}], "
                f"vertexShader: {vertex_shader}, fragmentShader: {fragment_shader}"
            )
            self.program = 0

    def _load_program(self, vertex_shader: int, fragment_shader: int) -> int:
        program = glCreateProgram()
        glAttachShader(program, vertex_shader)
        glAttachShader(program, fragment_shader)
        glLinkProgram(program)
        return program

    def get_attribute_location(self, attribute_name: str) -> int:
        if self.program != 0:
            return glGetAttribLocation(self.program, attribute_name)
        return -1

    def get_uniform_location(self, uniform_name: str) -> int:
        if self.program != 0:
            return glGetUniformLocation(self.program, uniform_name)
        return -1
