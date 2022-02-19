file(REMOVE_RECURSE
  "*"
  "App"
  "CMakeDoxyfile.in"
  "CMakeDoxygenDefaults.cmake"
  "Doxyfile"
  "Thirdparty"
  "framework"
  "hw"
  "util
"
  "CMakeCache.txt"
  "CMakeFiles/erase_ocd"
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
  include(CMakeFiles/erase_ocd.dir/cmake_clean_${lang}.cmake OPTIONAL)
endforeach()
