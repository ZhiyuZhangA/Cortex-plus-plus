#ifndef NOCOPYABLE_H
#define NOCOPYABLE_H

namespace cortex {
    class NonCopyable {
    public:
        NonCopyable() = default;
        NonCopyable(const NonCopyable&) = delete;
        NonCopyable & operator=(const NonCopyable&) = delete;
    };
}

#endif
