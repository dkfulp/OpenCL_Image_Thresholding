#include "xcl2.hpp"
#include <vector>
#include <fstream>
#include <iostream>
#include <string>
#include <cstdlib> // for exit()

using std::vector;

static const std::string error_message =
    "Error: Result mismatch:\n"
    "i = %d CPU result = %d Device result = %d\n";


// Function to take care of thresholding
int thresholding(int value, int maximum, int threshold){
    if (value > threshold){
        return 1;
    } else {
        return 0;
    }
}

// This example illustrates the very simple OpenCL example that performs simple image thresholding on a PGM image
int main(int argc, char **argv) {

    // Check for the correct number of arguements
    if (argc != 2) {
        std::cout << "Usage: " << argv[0] << " <XCLBIN File>" << std::endl;
        return EXIT_FAILURE;
    }

    // Set initial parameters
    std::string binaryFile = argv[1];
    cl_int err;
    cl::CommandQueue q;
    cl::Kernel krnl_threshold;
    cl::Context context;

    // DATA SETUP
    // #################

    // Read in file 
    std::ifstream inf;
    inf.open("coins.ascii.pgm", std::ios::in);
 
    // Check the file is open correctly
    if (!inf){
        // Print an error and exit
        std::cerr << "Error Opening Input File" << std::endl;
        exit(1);
    }

    // Get metadata information
    std::string file_type;
    getline(inf, file_type);
    std::string comments;
    getline(inf, comments);
    int width;
    inf >> width;
    int length;
    inf >> length;
    int maximum;
    inf >> maximum;

    // Get data size in bytes
    int data_size = width * length;
    size_t size_in_bytes = data_size * sizeof(int);

    // Reading in data values 
    int data [data_size];
    int num;
    int count = 0;
    while (inf >> num){
        data[count] = num;
        count++;
    }
    inf.close();

    // Setup output data array
    int out_data_serial[data_size];
    int out_data_parallel[data_size];

    // The get_xil_devices will return vector of Xilinx Devices
    auto devices = xcl::get_xil_devices();

    // read_binary_file() is a utility API which will load the binaryFile
    // and will return the pointer to file buffer.
    auto fileBuf = xcl::read_binary_file(binaryFile);
    cl::Program::Binaries bins{{fileBuf.data(), fileBuf.size()}};
    int valid_device = 0;
    for (unsigned int i = 0; i < devices.size(); i++) {
        auto device = devices[i];
        // Creating Context and Command Queue for selected Device
        OCL_CHECK(err, context = cl::Context({device}, NULL, NULL, NULL, &err));
        OCL_CHECK(err,
                  q = cl::CommandQueue(
                      context, {device}, CL_QUEUE_PROFILING_ENABLE, &err));

        std::cout << "Trying to program device[" << i
                  << "]: " << device.getInfo<CL_DEVICE_NAME>() << std::endl;
        cl::Program program(context, {device}, bins, NULL, &err);
        if (err != CL_SUCCESS) {
            std::cout << "Failed to program device[" << i
                      << "] with xclbin file!\n";
        } else {
            std::cout << "Device[" << i << "]: program successful!\n";
            // This call will extract a kernel out of the program we loaded in the
            // previous line. A kernel is an OpenCL function that is executed on the
            // FPGA. This function is defined in the src/thresholder.cl file.
            OCL_CHECK(
                err, krnl_threshold = cl::Kernel(program, "threshold", &err));
            valid_device++;
            break; // we break because we found a valid device
        }
    }
    if (valid_device == 0) {
        std::cout << "Failed to program any device found, exit!\n";
        exit(EXIT_FAILURE);
    }

    // These commands will allocate memory on the FPGA. The cl::Buffer objects can
    // be used to reference the memory locations on the device. The cl::Buffer
    // object cannot be referenced directly and must be passed to other OpenCL
    // functions.
    OCL_CHECK(err,
              cl::Buffer buffer_a(context,
                                  CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY,
                                  size_in_bytes,
                                  data,
                                  &err));

    OCL_CHECK(err,
              cl::Buffer buffer_result(context,
                                       CL_MEM_USE_HOST_PTR | CL_MEM_WRITE_ONLY,
                                       size_in_bytes,
                                       out_data_parallel,
                                       &err));

    //set the kernel Arguments
    int narg = 0;
    OCL_CHECK(err, err = krnl_threshold.setArg(narg++, buffer_result));
    OCL_CHECK(err, err = krnl_threshold.setArg(narg++, buffer_a));
    OCL_CHECK(err, err = krnl_threshold.setArg(narg++, data_size));

    // These commands will load the source_a and source_b vectors from the host
    // application and into the buffer_a and buffer_b cl::Buffer objects. The data
    // will be be transferred from system memory over PCIe to the FPGA on-board
    // DDR memory.
    OCL_CHECK(err,
              err = q.enqueueMigrateMemObjects({buffer_a},
                                               0 /* 0 means from host*/));

    //Launch the Kernel
    //This is equivalent to calling the enqueueNDRangeKernel function with a 
    //dimensionality of 1 and a global work size (NDRange) of 1.
    OCL_CHECK(err, err = q.enqueueTask(krnl_threshold));

    // The result of the previous kernel execution will need to be retrieved in
    // order to view the results. This call will write the data from the
    // buffer_result cl_mem object to the source_results vector
    OCL_CHECK(err,
              err = q.enqueueMigrateMemObjects({buffer_result},
                                               CL_MIGRATE_MEM_OBJECT_HOST));
    q.finish();

    // Open output file 
    std::ofstream outf;
    outf.open("parallel_coins.ascii.pgm", std::ios::out);
 
    // Check the file is open correctly
    if (!outf){
        // Print an error and exit
        std::cerr << "Error Opening Output File" << std::endl;
        exit(1);
    }

    // Write out metadata
    outf << file_type << std::endl;
    outf << comments << std::endl;
    outf << width << " " << length << std::endl;
    outf << 1 << std::endl;

    // Write out data to a file
    for (int i = 0; i < data_size; i++){
        outf << out_data_parallel[i] << " ";
    }
    outf.close();


    // SERIAL APPROACH
    // ##################

    for (int i = 0; i < data_size; i++){
        out_data_serial[i] = thresholding(data[i], maximum, 90);
    }

    // Read in file 
    outf.open("serial_coins.ascii.pgm", std::ios::out);
 
    // Check the file is open correctly
    if (!outf){
        // Print an error and exit
        std::cerr << "Error Opening Output File" << std::endl;
        exit(1);
    }

    // Write out metadata
    outf << file_type << std::endl;
    outf << comments << std::endl;
    outf << width << " " << length << std::endl;
    outf << 1 << std::endl;

    // Write out data to a file
    for (int i = 0; i < data_size; i++){
        outf << out_data_serial[i] << " ";
    }
    outf.close();

    int match = 0;
    for (int i = 0; i < data_size; i++){
        if (out_data_serial[i] != out_data_parallel[i]){
            match = 1;
            break;
        }
    }
 
    std::cout << "TEST " << (match ? "FAILED" : "PASSED") << std::endl;
    return (match ? EXIT_FAILURE : EXIT_SUCCESS);
}
