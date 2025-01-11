#ifndef NOCOPYABLE_H
#define NOCOPYABLE_H

namespace dl_core {
    class NonCopyable {
    public:
        NonCopyable() = default;
        NonCopyable(const NonCopyable&) = delete;
        NonCopyable & operator=(const NonCopyable&) = delete;
    };
}

#endif
