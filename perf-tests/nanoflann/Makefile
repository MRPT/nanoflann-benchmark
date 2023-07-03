# Set up basic variables:
CC         = g++
CFLAGS     = -c -Wall -O2 -mtune=native 
LDFLAGS    =

# List of sources:
SOURCES1    = test_nanoflann.cpp
OBJECTS1    = $(SOURCES1:.cpp=.o)

SOURCES2    = test_leaf_max_size.cpp
OBJECTS2    = $(SOURCES2:.cpp=.o)

SOURCES3    = test_n_thread_build.cpp
OBJECTS3    = $(SOURCES3:.cpp=.o)

# Name of executable target:
EXECUTABLE1 = test_nanoflann 

EXECUTABLE2 = test_leaf_max_size

EXECUTABLE3 = test_n_thread_build

# MRPT specific flags:
#  Here we invoke "pkg-config" passing it as argument the list of the 
#  MRPT libraries needed by our program (see available libs 
#   with "pkg-config --list-all | grep mrpt").
#
#CFLAGS     += `pkg-config --cflags mrpt-base`


all: $(SOURCES) $(EXECUTABLE1) $(EXECUTABLE2) $(EXECUTABLE3)
	
$(EXECUTABLE1): $(OBJECTS1)
	$(CC) $(LDFLAGS) $(OBJECTS1) -o $@

$(EXECUTABLE2): $(OBJECTS2)
	$(CC) $(LDFLAGS) $(OBJECTS2) -o $@

$(EXECUTABLE3): $(OBJECTS3)
	$(CC) $(LDFLAGS) $(OBJECTS3) -o $@

.cpp.o:
	$(CC) $(CFLAGS) $< -o $@

clean:
	rm *.o $(EXECUTABLE1) $(EXECUTABLE2) $(EXECUTABLE3)
