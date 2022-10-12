# build the library first in the ./lib folder
g++ -shared -fPIC -std=c++11 -I./pybind11/include/ `python3.10 -m pybind11 --includes` pybind.cpp -o mymodule.so `python3.10-config --ldflags`
# g++ -shared -fPIC -std=c++11 -I./pybind11/include/ `python3.10 -m pybind11 --includes` pybind.cpp -o mymodule.so `python3.10-config --ldflags`
# Then build our engine executable
# g++ -std=c++17 `python3.10 -m pybind11 --includes` engine.cpp -o engine `python3.6-config --ldflags`
