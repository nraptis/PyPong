# ShaderProgramShape2D.py

from graphics.shader_program import ShaderProgram
import ctypes

class ShaderProgramShape2D(ShaderProgram):
    def __init__(self, name: str, vertex_shader: int, fragment_shader: int):
        super().__init__(name, vertex_shader, fragment_shader)

        # Attribute locations
        self.attribute_location_position = self.get_attribute_location("Positions")

        # Uniform locations
        self.uniform_location_modulate_color = self.get_uniform_location("ModulateColor")
        self.uniform_location_projection_matrix = self.get_uniform_location("ProjectionMatrix")
        self.uniform_location_model_view_matrix = self.get_uniform_location("ModelViewMatrix")
        
        print(f"===> {name} ... attribute_location_position = {self.attribute_location_position}")
        print(f"===> {name} ... uniform_location_modulate_color = {self.uniform_location_modulate_color}")
        print(f"===> {name} ... uniform_location_projection_matrix = {self.uniform_location_projection_matrix}")
        print(f"===> {name} ... uniform_location_model_view_matrix = {self.uniform_location_model_view_matrix}")

        float_size = 4  # bytes per float

        self.attribute_stride_position = float_size * 2
        self.attribute_size_position = 2
        self.attribute_offset_position = ctypes.c_void_p(0)