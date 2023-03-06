#!/bin/bash

if [ "$1" = "" ]; then
    echo "$0: syntax is $0 dir"
    echo "    runs graphwords2.py on *.txt in dir for modes:"
    echo "      constituency, dependency, classic=4"
    exit 1
fi

DIR=$1
MODES="con dep classic=1"
FILES=`ls ${DIR}/*.txt`

for f in $FILES ; do
    BF=`basename $f`
    DN=`dirname $f`
    F=${DN}/${BF}
    echo "$0: working on $F"
    for m in $MODES ; do
	echo "  mode $m"
	time ./graphwords2.py $F $m 2>&1 | grep -v T5TokenizerFast | grep -v UserWarning | grep -v arg_constraints
    done
    DF=`basename $f .txt`
    mv ${DF}-*.dot ${DN}/
done


	 
