#define WIN32_LEAN_AND_MEAN
#include <Windows.h>

extern "C" {
    __declspec(dllexport) DWORD NvOptimusEnablement = 0x00000001;               // Prefer NVIDIA GPU
    __declspec(dllexport) DWORD AmdPowerXpressRequestHighPerformance = 0x00000001; // Prefer AMD GPU
}

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <GL/glew.h>
#include <GLFW/glfw3.h>

#include <iostream>

#include "rendering/RenderSurface.h"
#include "rendering/Camera.h"
#include "rendering/SimDraw.h"
#include "simulation/NBodySim.h"
#include "test/tests.h"

GLFWwindow* initWindow(int width, int height) {
    glfwInit();
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    glfwWindowHint(GLFW_RESIZABLE, GL_FALSE);

    GLFWwindow* window = glfwCreateWindow(width, height, "CUDA Raytracing", nullptr, nullptr);
    glfwMakeContextCurrent(window);
    glewInit();
    glViewport(0, 0, width, height);
    glClearColor(0.8f, 0.8f, 1.0f, 1.0f);
    return window;
}

int main()
{
    std::cout << "Test prefiSum: " << (testPrefixSum() ? "PASSED" : "FAILED") << "\n";
    std::cout << "Test splitAndSort: " << (testSplitAndSort() ? "PASSED" : "FAILED") << "\n";
    std::cout << "Test compact: " << (testCompact() ? "PASSED" : "FAILED") << "\n";
    std::cout << "Test radix sort: " << (testRadixSort() ? "PASSED" : "FAILED") << "\n";

    int width = 720;
    int height = 720;
    GLFWwindow* window = initWindow(width, height);
    cudaSetDevice(0);

    RenderSurface renderSurface(width, height);

    const GLubyte* r = glGetString(GL_RENDERER);
    std::cout << "GPU: " << r << std::endl;

    Camera cam = Camera(float3(500, 1200, 500), float3(0, -1, 0), float3(0, 0, 1), 1.0f);
    NBodySim sim = NBodySim(1000);
    SimDraw drawer = SimDraw(width, height, cam, sim.GetBodyInfos(), 1000, sim.GetCells(), sim.GetCellCount());

    while (!glfwWindowShouldClose(window)) {
        uchar4* devPtr = renderSurface.MapCudaResource();

        sim.Simulate();
        drawer.Render(devPtr);

        renderSurface.UnmapCudaResource();
        glClear(GL_COLOR_BUFFER_BIT);
        renderSurface.Draw();
        
        glfwSwapBuffers(window);
        glfwPollEvents();
    }

    renderSurface.Cleanup();
    glfwDestroyWindow(window);
    glfwTerminate();

    return 0;
}
