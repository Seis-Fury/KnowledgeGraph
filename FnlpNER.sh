#!/bin/bash

usage() {
	echo "Usage: $0 filename" >&2
}
if [ $# -gt 1 ]; then
	usage
fi

ARGS="-keepAllWhitespaces false"
if [ $# -eq 1 ];then
	input=$(cat $1)
fi

BASEDIR=/Users/rich/Arch/fnlp
JAVACMD="java -Xmx1024m -Dfile.encoding=UTF-8 -classpath ${BASEDIR}/fnlp-core/target/fnlp-core-2.1-SNAPSHOT.jar:${BASEDIR}/libs/trove4j-3.0.3.jar:${BASEDIR}/libs/commons-cli-1.2.jar org.fnlp.nlp.cn.tag.NERTagger -s ${BASEDIR}/models/seg.m ${BASEDIR}/models/pos.m "
${JAVACMD} "$input" 2>&1

