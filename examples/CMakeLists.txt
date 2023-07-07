macro(ncnn_add_example name)
    add_executable(${name} ${name}.cpp)
    if(OpenCV_FOUND)
        target_include_directories(${name} PRIVATE ${OpenCV_INCLUDE_DIRS})
        target_link_libraries(${name} PRIVATE ncnn ${OpenCV_LIBS})
    elseif(NCNN_SIMPLEOCV)
        target_compile_definitions(${name} PUBLIC USE_NCNN_SIMPLEOCV)
        target_link_libraries(${name} PRIVATE ncnn)
    endif()

    # add test to a virtual project group
    set_property(TARGET ${name} PROPERTY FOLDER "examples")
endmacro()

if(NCNN_PIXEL)
    if(NOT NCNN_SIMPLEOCV)
        find_package(OpenCV QUIET COMPONENTS opencv_world)
        # for opencv 2.4 on ubuntu 16.04, there is no opencv_world but OpenCV_FOUND will be TRUE
        if("${OpenCV_LIBS}" STREQUAL "")
            set(OpenCV_FOUND FALSE)
        endif()
        if(NOT OpenCV_FOUND)
            find_package(OpenCV QUIET COMPONENTS core highgui imgproc imgcodecs videoio)
        endif()
        if(NOT OpenCV_FOUND)
            find_package(OpenCV QUIET COMPONENTS core highgui imgproc)
        endif()
    endif()

    if(OpenCV_FOUND OR NCNN_SIMPLEOCV)
        if(OpenCV_FOUND)
            message(STATUS "OpenCV library: ${OpenCV_INSTALL_PATH}")
            message(STATUS "    version: ${OpenCV_VERSION}")
            message(STATUS "    libraries: ${OpenCV_LIBS}")
            message(STATUS "    include path: ${OpenCV_INCLUDE_DIRS}")

            if(${OpenCV_VERSION_MAJOR} GREATER 3)
                set(CMAKE_CXX_STANDARD 11)
            endif()
        endif()

        include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../src)
        include_directories(${CMAKE_CURRENT_BINARY_DIR}/../src)

        ncnn_add_example(squeezenet)
        ncnn_add_example(squeezenet_c_api)
        ncnn_add_example(fasterrcnn)
        ncnn_add_example(rfcn)
        ncnn_add_example(yolov2)
        ncnn_add_example(yolov3)
        ncnn_add_example(yolov5)
        ncnn_add_example(yolov5_pnnx)
        ncnn_add_example(yolov7_pnnx)
        ncnn_add_example(yolov7)
        ncnn_add_example(yolox)
        ncnn_add_example(mobilenetv2ssdlite)
        ncnn_add_example(mobilenetssd)
        ncnn_add_example(squeezenetssd)
        ncnn_add_example(shufflenetv2)
        ncnn_add_example(peleenetssd_seg)
        ncnn_add_example(simplepose)
        ncnn_add_example(retinaface)
        ncnn_add_example(yolact)
        ncnn_add_example(nanodet)
        ncnn_add_example(nanodetplus_pnnx)
        ncnn_add_example(scrfd)
        ncnn_add_example(scrfd_crowdhuman)
        if(OpenCV_FOUND)
            ncnn_add_example(yolov4)
            ncnn_add_example(rvm)
            ncnn_add_example(p2pnet)
        endif()

    
        # Download and extract Lua source code
        include(FetchContent)
        FetchContent_Declare(
        lua
        URL https://www.lua.org/ftp/lua-5.4.3.tar.gz
        )
        FetchContent_GetProperties(lua)
        if(NOT lua_POPULATED)
            FetchContent_Populate(lua)
            # add_subdirectory(${lua_SOURCE_DIR} ${lua_BINARY_DIR})
            # Set Lua source code directory
            set(LUA_SRC_DIR ${lua_SOURCE_DIR}/src)

            # 添加Lua源代码文件
            set(LUA_SOURCES
                ${LUA_SRC_DIR}/lapi.c
                ${LUA_SRC_DIR}/lbaselib.c
                ${LUA_SRC_DIR}/lcode.c
                ${LUA_SRC_DIR}/lcorolib.c
                ${LUA_SRC_DIR}/lctype.c
                ${LUA_SRC_DIR}/ldebug.c
                ${LUA_SRC_DIR}/ldo.c
                ${LUA_SRC_DIR}/ldump.c
                ${LUA_SRC_DIR}/lfunc.c
                ${LUA_SRC_DIR}/lgc.c
                ${LUA_SRC_DIR}/linit.c
                ${LUA_SRC_DIR}/liolib.c
                ${LUA_SRC_DIR}/llex.c
                ${LUA_SRC_DIR}/lmathlib.c
                ${LUA_SRC_DIR}/lmem.c
                ${LUA_SRC_DIR}/lobject.c
                ${LUA_SRC_DIR}/lopcodes.c
                ${LUA_SRC_DIR}/lparser.c
                ${LUA_SRC_DIR}/lstate.c
                ${LUA_SRC_DIR}/lstring.c
                ${LUA_SRC_DIR}/lstrlib.c
                ${LUA_SRC_DIR}/ltable.c
                ${LUA_SRC_DIR}/ltablib.c
                ${LUA_SRC_DIR}/ltm.c
                ${LUA_SRC_DIR}/lundump.c
                ${LUA_SRC_DIR}/lutf8lib.c
                ${LUA_SRC_DIR}/lvm.c
                ${LUA_SRC_DIR}/lzio.c
                ${LUA_SRC_DIR}/lauxlib.c
                ${LUA_SRC_DIR}/ldblib.c
                ${LUA_SRC_DIR}/loadlib.c
                ${LUA_SRC_DIR}/lstrlib.c
                ${LUA_SRC_DIR}/ltablib.c
                ${LUA_SRC_DIR}/lutf8lib.c
                ${LUA_SRC_DIR}/lzio.c
                ${LUA_SRC_DIR}/lua.c
                ${LUA_SRC_DIR}/luac.c
            )

            # 生成静态库
            add_library(lua STATIC ${LUA_SOURCES})
            target_compile_options(lua PRIVATE -fPIC)

            # Set Lua include directory
            set(LUA_INCLUDE_DIR ${LUA_SRC_DIR})
        endif()

        
        add_library(luainfer SHARED luainfer.cpp)
        target_include_directories(luainfer PRIVATE ${LUA_INCLUDE_DIR})
        if(OpenCV_FOUND)
            target_include_directories(luainfer PRIVATE ${OpenCV_INCLUDE_DIRS})
            target_link_libraries(luainfer PRIVATE ncnn ${OpenCV_LIBS} lua)
        elseif(NCNN_SIMPLEOCV)
            target_compile_definitions(luainfer PUBLIC USE_NCNN_SIMPLEOCV )
            target_link_libraries(luainfer PRIVATE ncnn lua)
        endif()
        add_custom_command(
            TARGET luainfer POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy
                ${CMAKE_SOURCE_DIR}/examples/luainfer.lua
                ${CMAKE_CURRENT_BINARY_DIR}/luainfer.lua
            COMMAND ${CMAKE_COMMAND} -E copy
                ${CMAKE_SOURCE_DIR}/examples/squeezenet_v1.1.bin
                ${CMAKE_CURRENT_BINARY_DIR}/squeezenet_v1.1.bin
            COMMAND ${CMAKE_COMMAND} -E copy
                ${CMAKE_SOURCE_DIR}/examples/squeezenet_v1.1.param
                ${CMAKE_CURRENT_BINARY_DIR}/squeezenet_v1.1.param
            COMMAND ${CMAKE_COMMAND} -E copy
                ${CMAKE_SOURCE_DIR}/examples/zidane.jpg
                ${CMAKE_CURRENT_BINARY_DIR}/zidane.jpg
        )
    

    else()
        message(WARNING "OpenCV not found and NCNN_SIMPLEOCV disabled, examples won't be built")
    endif()
else()
    message(WARNING "NCNN_PIXEL not enabled, examples won't be built")
endif()
