# PyPong

> You have a basic understanding of a language once you can build Pong in the language.

PyPong is a small OpenGL Pong clone written in pure Python.  
It uses a tiny “engine” layer (`graphics_library.py`, `graphics_matrix.py`, etc.) on top of:

- **PyOpenGL** (OpenGL bindings)
- **glfw** (window + input)
- **NumPy** (array + math utilities)
- **Pillow** (image loading for textures)

---

## Requirements

- **Python**: 3.10+ recommended  
- **Platform**: macOS, Linux, or Windows with an OpenGL 2.1+ compatible GPU  
- **Python libraries**:
  - `numpy`
  - `Pillow` (provides `PIL`)
  - `PyOpenGL`
  - `glfw`

Standard library modules like `sys`, `ctypes`, `enum`, `dataclasses`, `time`, `random`, etc. are used as well, but you don’t need to install anything extra for those.

---

## Quick Start

### 1. Clone the repo

```bash
git clone https://github.com/nraptis/PyPong.git
cd PyPong


python -m venv .venv
.venv\Scripts\activate
pip install numpy Pillow PyOpenGL glfw
python main.py
