Python Project Structure
====================================
There are two main general structures: the `flat layout` and the `src layout`.


flat-layout:
============
<pre>
├── README.md
├── noxfile.py
├── pyproject.toml
├── setup.py
├── awesome_package/
│   ├── __init__.py
│   └── module.py
└── tools/
    ├── generate_awesomeness.py
    └── decrease_world_suck.py
</pre>


src-layout:
=============
<pre>
├── README.md
├── noxfile.py
├── pyproject.toml
├── setup.py
├── src/
│    └── awesome_package/
│       ├── __init__.py
│       └── module.py
└── tools/
    ├── generate_awesomeness.py
    └── decrease_world_suck.py
</pre>

References
==============
- [src-layout-vs-flat-layout](https://packaging.python.org/en/latest/discussions/src-layout-vs-flat-layout/)
- [a-practical-guide-to-python-project-structure-and-packaging](https://medium.com/mlearning-ai/a-practical-guide-to-python-project-structure-and-packaging-90c7f7a04f95)
- [package_discovery](https://setuptools.pypa.io/en/latest/userguide/package_discovery.html)