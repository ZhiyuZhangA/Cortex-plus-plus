#include "DLEngine/DLEngine.h"

namespace cortex_core {

    bool DLEngine::m_grad_mode_enabled = true;

    void DLEngine::enable_grad() {
        m_grad_mode_enabled = true;
    }

    void DLEngine::inference_mode() {
        m_grad_mode_enabled = false;
    }

    bool DLEngine::is_grad_mode() {
        return m_grad_mode_enabled;
    }
}
