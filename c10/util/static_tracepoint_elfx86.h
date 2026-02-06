#pragma once

// clang-format off

// Default constraint for the probe arguments as operands.
#ifndef SMITH_SDT_ARG_CONSTRAINT
#define SMITH_SDT_ARG_CONSTRAINT      "nor"
#endif

// Instruction to emit for the probe.
#define SMITH_SDT_NOP                 nop

// Note section properties.
#define SMITH_SDT_NOTE_NAME           "stapsdt"
#define SMITH_SDT_NOTE_TYPE           3

// Semaphore variables are put in this section
#define SMITH_SDT_SEMAPHORE_SECTION   ".probes"

// Size of address depending on platform.
#ifdef __LP64__
#define SMITH_SDT_ASM_ADDR            .8byte
#else
#define SMITH_SDT_ASM_ADDR            .4byte
#endif

// Assembler helper Macros.
#define SMITH_SDT_S(x)                #x
#define SMITH_SDT_ASM_1(x)            SMITH_SDT_S(x) "\n"
#define SMITH_SDT_ASM_2(a, b)         SMITH_SDT_S(a) "," SMITH_SDT_S(b) "\n"
#define SMITH_SDT_ASM_3(a, b, c)      SMITH_SDT_S(a) "," SMITH_SDT_S(b) ","    \
                                      SMITH_SDT_S(c) "\n"
#define SMITH_SDT_ASM_STRING(x)       SMITH_SDT_ASM_1(.asciz SMITH_SDT_S(x))

// Helper to determine the size of an argument.
#define SMITH_SDT_IS_ARRAY_POINTER(x)  ((__builtin_classify_type(x) == 14) ||  \
                                        (__builtin_classify_type(x) == 5))
#define SMITH_SDT_ARGSIZE(x)  (SMITH_SDT_IS_ARRAY_POINTER(x)                   \
                               ? sizeof(void*)                                 \
                               : sizeof(x))

// Format of each probe arguments as operand.
// Size of the argument tagged with SMITH_SDT_Sn, with "n" constraint.
// Value of the argument tagged with SMITH_SDT_An, with configured constraint.
#define SMITH_SDT_ARG(n, x)                                                    \
  [SMITH_SDT_S##n] "n"                ((size_t)SMITH_SDT_ARGSIZE(x)),          \
  [SMITH_SDT_A##n] SMITH_SDT_ARG_CONSTRAINT (x)

// Templates to append arguments as operands.
#define SMITH_SDT_OPERANDS_0()        [__sdt_dummy] "g" (0)
#define SMITH_SDT_OPERANDS_1(_1)      SMITH_SDT_ARG(1, _1)
#define SMITH_SDT_OPERANDS_2(_1, _2)                                           \
  SMITH_SDT_OPERANDS_1(_1), SMITH_SDT_ARG(2, _2)
#define SMITH_SDT_OPERANDS_3(_1, _2, _3)                                       \
  SMITH_SDT_OPERANDS_2(_1, _2), SMITH_SDT_ARG(3, _3)
#define SMITH_SDT_OPERANDS_4(_1, _2, _3, _4)                                   \
  SMITH_SDT_OPERANDS_3(_1, _2, _3), SMITH_SDT_ARG(4, _4)
#define SMITH_SDT_OPERANDS_5(_1, _2, _3, _4, _5)                               \
  SMITH_SDT_OPERANDS_4(_1, _2, _3, _4), SMITH_SDT_ARG(5, _5)
#define SMITH_SDT_OPERANDS_6(_1, _2, _3, _4, _5, _6)                           \
  SMITH_SDT_OPERANDS_5(_1, _2, _3, _4, _5), SMITH_SDT_ARG(6, _6)
#define SMITH_SDT_OPERANDS_7(_1, _2, _3, _4, _5, _6, _7)                       \
  SMITH_SDT_OPERANDS_6(_1, _2, _3, _4, _5, _6), SMITH_SDT_ARG(7, _7)
#define SMITH_SDT_OPERANDS_8(_1, _2, _3, _4, _5, _6, _7, _8)                   \
  SMITH_SDT_OPERANDS_7(_1, _2, _3, _4, _5, _6, _7), SMITH_SDT_ARG(8, _8)
#define SMITH_SDT_OPERANDS_9(_1, _2, _3, _4, _5, _6, _7, _8, _9)               \
  SMITH_SDT_OPERANDS_8(_1, _2, _3, _4, _5, _6, _7, _8), SMITH_SDT_ARG(9, _9)

// Templates to reference the arguments from operands in note section.
#define SMITH_SDT_ARGFMT(no)        %n[SMITH_SDT_S##no]@%[SMITH_SDT_A##no]
#define SMITH_SDT_ARG_TEMPLATE_0    /*No arguments*/
#define SMITH_SDT_ARG_TEMPLATE_1    SMITH_SDT_ARGFMT(1)
#define SMITH_SDT_ARG_TEMPLATE_2    SMITH_SDT_ARG_TEMPLATE_1 SMITH_SDT_ARGFMT(2)
#define SMITH_SDT_ARG_TEMPLATE_3    SMITH_SDT_ARG_TEMPLATE_2 SMITH_SDT_ARGFMT(3)
#define SMITH_SDT_ARG_TEMPLATE_4    SMITH_SDT_ARG_TEMPLATE_3 SMITH_SDT_ARGFMT(4)
#define SMITH_SDT_ARG_TEMPLATE_5    SMITH_SDT_ARG_TEMPLATE_4 SMITH_SDT_ARGFMT(5)
#define SMITH_SDT_ARG_TEMPLATE_6    SMITH_SDT_ARG_TEMPLATE_5 SMITH_SDT_ARGFMT(6)
#define SMITH_SDT_ARG_TEMPLATE_7    SMITH_SDT_ARG_TEMPLATE_6 SMITH_SDT_ARGFMT(7)
#define SMITH_SDT_ARG_TEMPLATE_8    SMITH_SDT_ARG_TEMPLATE_7 SMITH_SDT_ARGFMT(8)
#define SMITH_SDT_ARG_TEMPLATE_9    SMITH_SDT_ARG_TEMPLATE_8 SMITH_SDT_ARGFMT(9)

// Resolvable by name macros
// An attribute that marks a function or variable as needing to be resolvable
// by name. This generally is needed if inline assembly refers to the variable
// by string name.
#ifdef __roar__
#define SMITH_NAME_RESOLVABLE __attribute__((roar_resolvable_by_name))
#else
#define SMITH_NAME_RESOLVABLE
#endif

// Semaphore define, declare and probe note format

#define SMITH_SDT_SEMAPHORE(provider, name)                                    \
  smith_sdt_semaphore_##provider##_##name

#define SMITH_SDT_DEFINE_SEMAPHORE(name)                                       \
  extern "C" {                                                                 \
    SMITH_NAME_RESOLVABLE                                                      \
    volatile unsigned short SMITH_SDT_SEMAPHORE(blacksmith, name)                 \
    __attribute__((section(SMITH_SDT_SEMAPHORE_SECTION), used)) = 0;           \
  }

#define SMITH_SDT_DECLARE_SEMAPHORE(name)                                      \
  extern "C" SMITH_NAME_RESOLVABLE volatile unsigned short                     \
    SMITH_SDT_SEMAPHORE(blacksmith, name)

#define SMITH_SDT_SEMAPHORE_NOTE_0(provider, name)                             \
  SMITH_SDT_ASM_1(     SMITH_SDT_ASM_ADDR 0) /*No Semaphore*/                  \

#define SMITH_SDT_SEMAPHORE_NOTE_1(provider, name)                             \
  SMITH_SDT_ASM_1(SMITH_SDT_ASM_ADDR SMITH_SDT_SEMAPHORE(provider, name))

// Structure of note section for the probe.
#define SMITH_SDT_NOTE_CONTENT(provider, name, has_semaphore, arg_template)    \
  SMITH_SDT_ASM_1(990: SMITH_SDT_NOP)                                          \
  SMITH_SDT_ASM_3(     .pushsection .note.stapsdt,"","note")                   \
  SMITH_SDT_ASM_1(     .balign 4)                                              \
  SMITH_SDT_ASM_3(     .4byte 992f-991f, 994f-993f, SMITH_SDT_NOTE_TYPE)       \
  SMITH_SDT_ASM_1(991: .asciz SMITH_SDT_NOTE_NAME)                             \
  SMITH_SDT_ASM_1(992: .balign 4)                                              \
  SMITH_SDT_ASM_1(993: SMITH_SDT_ASM_ADDR 990b)                                \
  SMITH_SDT_ASM_1(     SMITH_SDT_ASM_ADDR 0) /*Reserved for Base Address*/     \
  SMITH_SDT_SEMAPHORE_NOTE_##has_semaphore(provider, name)                     \
  SMITH_SDT_ASM_STRING(provider)                                               \
  SMITH_SDT_ASM_STRING(name)                                                   \
  SMITH_SDT_ASM_STRING(arg_template)                                           \
  SMITH_SDT_ASM_1(994: .balign 4)                                              \
  SMITH_SDT_ASM_1(     .popsection)

// Main probe Macro.
#define SMITH_SDT_PROBE(provider, name, has_semaphore, n, arglist)             \
    __asm__ __volatile__ (                                                     \
      SMITH_SDT_NOTE_CONTENT(                                                  \
        provider, name, has_semaphore, SMITH_SDT_ARG_TEMPLATE_##n)             \
      :: SMITH_SDT_OPERANDS_##n arglist                                        \
    )                                                                          \

// Helper Macros to handle variadic arguments.
#define SMITH_SDT_NARG_(_0, _1, _2, _3, _4, _5, _6, _7, _8, _9, N, ...) N
#define SMITH_SDT_NARG(...)                                                    \
  SMITH_SDT_NARG_(__VA_ARGS__, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0)
#define SMITH_SDT_PROBE_N(provider, name, has_semaphore, N, ...)               \
  SMITH_SDT_PROBE(provider, name, has_semaphore, N, (__VA_ARGS__))
