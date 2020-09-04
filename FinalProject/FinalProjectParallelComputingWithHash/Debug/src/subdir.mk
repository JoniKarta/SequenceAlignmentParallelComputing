################################################################################
# Automatically-generated file. Do not edit!
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
C_SRCS += \
../src/finalProjectParallelComputing.c \
../src/utility.c 

O_SRCS += \
../src/cudaFunctions.o \
../src/finalProjectParallelComputing.o \
../src/utility.o 

OBJS += \
./src/finalProjectParallelComputing.o \
./src/utility.o 

C_DEPS += \
./src/finalProjectParallelComputing.d \
./src/utility.d 


# Each subdirectory must supply rules for building sources it contributes
src/%.o: ../src/%.c
	@echo 'Building file: $<'
	@echo 'Invoking: GCC C Compiler'
	mpicc -fopenmp -O0 -g3 -Wall -c -fmessage-length=0 -MMD -MP -MF"$(@:%.o=%.d)" -MT"$(@)" -o "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '


