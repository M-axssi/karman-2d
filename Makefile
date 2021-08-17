SHELL:=/bin/bash

#---------------------------------------------------------
#ref-data
Ref_Data:karman-2d.py
	for i in `seq 0 5`; do \
		python $^ -o Reference-Data -r 128 -l 100 --re `echo $$(( 10000 * 2**($$i+4) ))`; \
	done

#---------------------------------------------------------
#test-data
Test_Data:karman-2d.py
	for i in `seq 0 5`; do \
		python $^ -o Test-Data -r 128 -l 100 --re `echo $$(( 10000 * 2**($$i+3)*3 ))`; \
	done


#---------------------------------------------------------
#pre-data
Pre_Data:karman-2d-pre.py
	for i in `seq 0 5`; do \
		python $^ -o Pre-Data -r 32 --beta 0 --scale 4 -l 100 --re `echo $$(( 10000 * 2**($$i+4) ))`; \
	done

#---------------------------------------------------------
#pre-data for pre_sr
Pre_Data_Sr:karman-2d-pre.py
	for i in `seq 0 5`; do \
		python $^ -o Pre-Data-Sr -r 32 --beta 1 --scale 4 -l 100 --re `echo $$(( 10000 * 2**($$i+4) ))`; \
	done

#---------------------------------------------------------
#train NON
NON:karman-2d-train-NON.py
	python $^ -r 32 -l 100 --device CPU --steps 500 --initial_step 1000 --epoch 100 --input=./Reference-Data

#---------------------------------------------------------
#train PRO_NON
PRO_NON:karman-2d-train-PRO-NON.py
	python $^ -r 32 -l 100 --device CPU --steps 500 --initial_step 1000 --epoch 100 --with_steps 4 \
	--output ./model/PRO_NON_4.pth --input=./Reference-Data

#---------------------------------------------------------
#train PRE
PRE:karman-2d-train-PRE.py
	python $^ -r 32 -l 100 --device CPU --steps 500 --initial_step 1000 --epoch 500 --input ./Pre-Data

#---------------------------------------------------------
#train SOL
SOL:karman-2d-train-SOL.py
	python $^ -r 32 -l 100 --device CPU --steps 500 --initial_step 1000 --epoch 100 --with_steps 4 \
	--output ./model/SOL_4.pth --input=./Reference-Data

#---------------------------------------------------------
#Test source data
Source_Test:karman-2d-source-test.py
	python $^ -r 32 -l 100 -s 5 --initial_step 1000 --input=./Test-Data

#---------------------------------------------------------
#Allpy
Apply:karman-2d-apply.py
	python $^ -r 32 -l 100 -s 5 --device CPU --initial_step 1000 --input=./Test-Data --model_path=./model/PRO_NON_8_36.pth

#---------------------------------------------------------
#Generate imga
Generate_Imag:karman-2d-show.py
	python $^ -r 32 -l 100 -s 5 --device CPU --initial_step 1000  --steps 200 --input=./Test-Data \
	--model_path=./model/NON.pth --output ./imag_result/SRC --type Source --mark 4