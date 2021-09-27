
SRC = src
BLD = pybuild


			

CXXFLAGS=-std=c++11

tocopy=setup.py boruvka_superpixel_wrap.pxd boruvka_superpixel_wrap.pyx \
       boruvka_superpixel.h boruvka_superpixel.cpp boruvkasupix.py

all: module

clean:
	rm -rf $(BLD)

builddir:
	mkdir -p $(BLD)

module: builddir
	cp $(addprefix $(SRC)/,$(tocopy)) $(BLD)
	cd $(BLD); python3 setup.py build_ext --inplace

$(BLD)/boruvka_superpixel.o: builddir \
		$(SRC)/boruvka_superpixel.h $(SRC)/boruvka_superpixel.cpp
	$(CXX) $(CXXFLAGS) -c $(SRC)/boruvka_superpixel.cpp -o $@

#$(BLD)/boruvkasupix: $(BLD)/boruvka_superpixel.o $(SRC)/main.cpp
#	$(CXX) $(CXXFLAGS) $^ -o $@
