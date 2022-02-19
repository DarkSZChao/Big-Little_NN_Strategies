OUTPUT_FORMAT ("elf32-littlearm", "elf32-bigarm", "elf32-littlearm")
ENTRY(Reset_Handler)
MEMORY
{
    FLASH (rx) : ORIGIN = 0x01000000, LENGTH = (0x80000 - 0x16000)
    DSP (rx) : ORIGIN = (0x01000000 + (0x80000 - 0x16000)), LENGTH = 0x16000
    RAM (rwx) : ORIGIN = 0x10000000, LENGTH = (0x40000 - (80000 + 0x200))
    SHM (rwx) : ORIGIN = (0x10040000 - 0x200), LENGTH = 0x200
    HEAP (rwx) : ORIGIN = (0x10040000 - (80000 + 0x200)), LENGTH = 80000
}
_estack = ORIGIN(RAM) + LENGTH(RAM) - 0x20;
SECTIONS
{
  .reset :
  {
    __reset_start = .;
        KEEP(*(.isr_vector))
        . = ALIGN(0x4);
    __reset_end = .;
  } >FLASH
  .ram_vectors (NOLOAD) : {
    __ram_vectors_start = .;
    * (.ram_vectors);
    . = ALIGN(4);
    __ram_vectors_end = .;
  } > RAM
  __ram_vectors_size = __ram_vectors_end - __ram_vectors_start;
  .text :
  {
    CREATE_OBJECT_SYMBOLS
        __text_load = LOADADDR(.text);
        __text_start = .;
        *(EXCLUDE_FILE(*eta_csp_buck*.o *eta_csp_dsp*.o *ssd_detection*.o
            *qfplib-m3*.o *stats*.o *printf-stdarg*.o *ecm3532_pinconfig*.o
            *board_config*.o *system_ecm3532*.o *uart_hal*.o *eta_csp_uart*.o
            *eta_csp_flash*.o *eta_csp*.o *libgcc.a*) .text* .rodata*)
        *crtbegin.o(.ctors)
        *crtbegin?.o(.ctors)
        *(EXCLUDE_FILE(*crtend?.o *crtend.o) .ctors)
        *(SORT(.ctors.*))
        *(.ctors)
        *crtbegin.o(.dtors)
        *crtbegin?.o(.dtors)
        *(EXCLUDE_FILE(*crtend?.o *crtend.o) .dtors)
        *(SORT(.dtors.*))
        *(.dtors)
        *(.rodata .rodata.* .gnu.linkonce.r.*)
        KEEP(*(.eh_frame*))
        _etext = .;
    } >RAM AT>FLASH
    .shmSectionBlock (NOLOAD):
    {
        __SHM_SECTION_START = .;
        KEEP(*(.shmSection))
        __SHM_SECTION_END = .;
    } > SHM
    .data :
    {
        __data_load = LOADADDR(.data);
        . = ALIGN(4);
        __data_start = .;
        *(vtable)
        *(.data*)
        PROVIDE_HIDDEN (__preinit_array_start = .);
        KEEP(*(.preinit_array))
        PROVIDE_HIDDEN (__preinit_array_end = .);
        . = ALIGN(4);
        PROVIDE_HIDDEN (__init_array_start = .);
        KEEP(*(SORT(.init_array.*)))
        KEEP(*(.init_array))
        PROVIDE_HIDDEN (__init_array_end = .);
        . = ALIGN(4);
        PROVIDE_HIDDEN (__fini_array_start = .);
        KEEP(*(SORT(.fini_array.*)))
        KEEP(*(.fini_array))
        PROVIDE_HIDDEN (__fini_array_end = .);
        KEEP(*(.jcr*))
        . = ALIGN(4);
     _edata = .;
    } >RAM AT>FLASH
   .bss :
    {
        . = ALIGN(4);
        __bss_start__ = .;
        *(.bss*)
        *(COMMON)
        . = ALIGN(4);
        __bss_end__ = .;
    } > RAM
    . = ALIGN(8);
    __HeapBase = .;
    end = .;
    _ram_start = ORIGIN(RAM);
    .stack_dummy (COPY):
    {
        *(.stack*)
    } > RAM
    __exidx_start = .;
    .ARM.exidx :
    {
        *(.ARM.exidx* .gnu.linkonce.armexidx.*)
    } > FLASH
    __exidx_end = .;
    .binSection :
    {
        __BIN_SECTION_START = .;
        KEEP(*(.binSection))
        __BIN_SECTION_END = .;
    } > FLASH
    .initSection :
    {
        KEEP(*(.init))
        KEEP(*(.fini))
        *eta_csp_buck*.o(.text* .rodata*)
        *eta_csp_dsp*.o(.text* .rodata*)
        *ssd_detection*.o(.text* .rodata*)
        *qfplib-m3*.o(.text* .rodata*)
        *stats*.o(.text* .rodata*)
        *arm_addsubsf3*.o(.text* .rodata*)
        *_arm_fixsfsi*.o(.text* .rodata*)
        *printf-stdarg*.o(.text* .rodata*)
        *ecm3532_pinconfig*.o(.text* .rodata*)
        *board_config*.o(.text* .rodata*)
        *system_ecm3532*.o(.text* .rodata*)
        *uart_hal*.o(.text* .rodata*)
        *eta_csp_uart*.o(.text* .rodata*)
        *eta_csp_flash*.o(.text* .rodata*)
        *eta_csp*.o(.text* .rodata*)
        */libgcc.a:*
        */libc.a:*
        */libm.a:*
    } > FLASH
    .appHeapBlock (NOLOAD) :
    {
        __heap_start__ = .;
        *(.appHeap*)
        __heap_end__ = .;
    } > HEAP
    .dspBlock :
    {
        KEEP(*(.dsp))
    } >DSP
    .pbufBlock :
    {
        KEEP(*(.pbuf))
    } >RAM AT>FLASH
    __StackTop = ORIGIN(RAM) + LENGTH(RAM) - 4;
    __StackLimit = __StackTop - SIZEOF(.stack_dummy) ;
    PROVIDE(__stack = __StackTop);
    __HeapLimit = __StackLimit;
    ASSERT(__HeapBase <= __HeapLimit, "region RAM overflowed with stack")
}
