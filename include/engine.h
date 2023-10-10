#pragma once

class Engine {
public:
    Engine() = default;
    virtual ~Engine() = default;
    virtual int init() = 0;
    virtual int infer(void* input, void* output) = 0;
    virtual int destroy() = 0;
};
