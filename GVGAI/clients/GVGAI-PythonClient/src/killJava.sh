#/bin/bash

JAVA_PROCESS=$(ps | grep "java" | awk '{print $1}')

if [ -z $JAVA_PROCESS ]
then
  echo "Java process not found. Exiting..."
  exit 1
else
  printf "Found Java process with the following PID: %d. Proceeding to kill it...\n" $JAVA_PROCESS
  kill -9 $JAVA_PROCESS
fi

