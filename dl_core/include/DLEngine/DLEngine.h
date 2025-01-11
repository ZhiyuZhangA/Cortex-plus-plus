#ifndef DLENGINE_H
#define DLENGINE_H

namespace cortex_core {
    class DLEngine {
    public:
        static void enable_grad();
        static void inference_mode();
        static bool is_grad_mode();

    private:
        static bool m_grad_mode_enabled;
    };
}

#endif
