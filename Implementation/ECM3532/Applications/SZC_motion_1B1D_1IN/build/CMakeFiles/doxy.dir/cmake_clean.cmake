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
  "CMakeFiles/doxy"
  "CPackConfig.cmake"
  "CPackSourceConfig.cmake"
  "\\*.bin"
  "\\*.lss"
  "\\*.map"
  "\\.cmake"
  "_CPack_Packages"
  "cmake_install.cmake"
  "doc_doxygen/html/index.html"
  "util"
)

# Per-language clean rules from dependency scanning.
foreach(lang )
  include(CMakeFiles/doxy.dir/cmake_clean_${lang}.cmake OPTIONAL)
endforeach()
