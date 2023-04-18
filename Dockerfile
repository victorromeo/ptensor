FROM alpine:3.14
RUN apk update && apk upgrade && apk add --no-cache cmake build-base libc-dev
COPY . /app
RUN cd /app && rm -rf build && mkdir build && cd build && cmake .. -DINSTALL_GTEST=OFF -DPACKAGE_TESTS=ON -DPACKAGE_TUTORIALS=ON && make && ctest && make install
WORKDIR /app/build
ENTRYPOINT [ "ctest" ]