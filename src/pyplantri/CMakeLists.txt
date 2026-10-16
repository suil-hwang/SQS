cmake_minimum_required(VERSION 3.15)
project(plantri C)

add_executable(plantri_sqs src/plantri_sqs.c)
target_compile_features(plantri_sqs PRIVATE c_std_99)
if(WIN32)
  target_include_directories(plantri_sqs PRIVATE src/compat)
endif()

install(TARGETS plantri_sqs RUNTIME DESTINATION bin)
