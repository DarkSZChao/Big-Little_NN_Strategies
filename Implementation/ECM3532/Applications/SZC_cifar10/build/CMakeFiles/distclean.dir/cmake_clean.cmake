file(REMOVE_RECURSE
  "*"
  "App"
  "CMakeDoxyfile.in"
  "CMakeDoxygenDefaults.cmake"
  "Doxyfile"
  "SZC_cifar10.elf.json"
  "Thirdparty"
  "dsp_fw-prefix"
  "dsp_src-prefix"
  "framework"
  "hw"
  "util
"
  "CMakeCache.txt"
  "CMakeFiles/distclean"
  "CPackConfig.cmake"
  "CPackSourceConfig.cmake"
  "\\*.bin"
  "\\*.lss"
  "\\*.map"
  "\\.cmake"
  "_CPack_Packages"
  "cmake_install.cmake"
  "util"
)

# Per-language clean rules from dependency scanning.
foreach(lang )
  include(CMakeFiles/distclean.dir/cmake_clean_${lang}.cmake OPTIONAL)
endforeach()
