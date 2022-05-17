alias nsmi='nvidia-smi'
alias wsmi='watch -n 1 nvidia-smi'
count (){
	if [ -d "$@" ]; then
	    echo "$(find "$@" -type f | wc -l) files, $(find "$@" -type d | wc -l) dirs"
	else
	    echo "[ERROR]  Please provide a directory."
	    exit 1
	fi
}
source activate nnunet_pathology
