е

щ
^
AssignVariableOp
resource
value"dtype"
dtypetype"
validate_shapebool( 

BiasAdd

value"T	
bias"T
output"T""
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
8
Const
output"dtype"
valuetensor"
dtypetype

Conv2D

input"T
filter"T
output"T"
Ttype:	
2"
strides	list(int)"
use_cudnn_on_gpubool(",
paddingstring:
SAMEVALIDEXPLICIT""
explicit_paddings	list(int)
 "-
data_formatstringNHWC:
NHWCNCHW" 
	dilations	list(int)

$
DisableCopyOnRead
resource
W

ExpandDims

input"T
dim"Tdim
output"T"	
Ttype"
Tdimtype0:
2	
.
Identity

input"T
output"T"	
Ttype
9
	IdentityN

input2T
output2T"
T
list(type)(0
u
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:
2	

MaxPool

input"T
output"T"
Ttype0:
2	"
ksize	list(int)(0"
strides	list(int)(0",
paddingstring:
SAMEVALIDEXPLICIT""
explicit_paddings	list(int)
 ":
data_formatstringNHWC:
NHWCNCHWNCHW_VECT_C

MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool("
allow_missing_filesbool( 
?
Mul
x"T
y"T
z"T"
Ttype:
2	

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
Г
PartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetype
[
Reshape
tensor"T
shape"Tshape
output"T"	
Ttype"
Tshapetype0:
2	
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0
?
Select
	condition

t"T
e"T
output"T"	
Ttype
d
Shape

input"T&
output"out_typeэout_type"	
Ttype"
out_typetype0:
2	
H
ShardedFilename
basename	
shard

num_shards
filename
0
Sigmoid
x"T
y"T"
Ttype:

2
N
Squeeze

input"T
output"T"	
Ttype"
squeeze_dims	list(int)
 (
С
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring Ј
@
StaticRegexFullMatch	
input

output
"
patternstring
ї
StridedSlice

input"T
begin"Index
end"Index
strides"Index
output"T"	
Ttype"
Indextype:
2	"

begin_maskint "
end_maskint "
ellipsis_maskint "
new_axis_maskint "
shrink_axis_maskint 
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 
P
	Transpose
x"T
perm"Tperm
y"T"	
Ttype"
Tpermtype0:
2	

VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 "serve*2.12.12v2.12.0-25-g8e2b6655c0c8і
^
countVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namecount
W
count/Read/ReadVariableOpReadVariableOpcount*
_output_shapes
: *
dtype0
^
totalVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nametotal
W
total/Read/ReadVariableOpReadVariableOptotal*
_output_shapes
: *
dtype0
b
count_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	count_1
[
count_1/Read/ReadVariableOpReadVariableOpcount_1*
_output_shapes
: *
dtype0
b
total_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	total_1
[
total_1/Read/ReadVariableOpReadVariableOptotal_1*
_output_shapes
: *
dtype0
n
Adam/v/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameAdam/v/bias
g
Adam/v/bias/Read/ReadVariableOpReadVariableOpAdam/v/bias*
_output_shapes
:*
dtype0
n
Adam/m/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameAdam/m/bias
g
Adam/m/bias/Read/ReadVariableOpReadVariableOpAdam/m/bias*
_output_shapes
:*
dtype0
v
Adam/v/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*
shared_nameAdam/v/kernel
o
!Adam/v/kernel/Read/ReadVariableOpReadVariableOpAdam/v/kernel*
_output_shapes

:*
dtype0
v
Adam/m/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*
shared_nameAdam/m/kernel
o
!Adam/m/kernel/Read/ReadVariableOpReadVariableOpAdam/m/kernel*
_output_shapes

:*
dtype0
r
Adam/v/bias_1VarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameAdam/v/bias_1
k
!Adam/v/bias_1/Read/ReadVariableOpReadVariableOpAdam/v/bias_1*
_output_shapes
:*
dtype0
r
Adam/m/bias_1VarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameAdam/m/bias_1
k
!Adam/m/bias_1/Read/ReadVariableOpReadVariableOpAdam/m/bias_1*
_output_shapes
:*
dtype0
~
Adam/v/kernel_1VarHandleOp*
_output_shapes
: *
dtype0*
shape:** 
shared_nameAdam/v/kernel_1
w
#Adam/v/kernel_1/Read/ReadVariableOpReadVariableOpAdam/v/kernel_1*"
_output_shapes
:**
dtype0
~
Adam/m/kernel_1VarHandleOp*
_output_shapes
: *
dtype0*
shape:** 
shared_nameAdam/m/kernel_1
w
#Adam/m/kernel_1/Read/ReadVariableOpReadVariableOpAdam/m/kernel_1*"
_output_shapes
:**
dtype0
n
learning_rateVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namelearning_rate
g
!learning_rate/Read/ReadVariableOpReadVariableOplearning_rate*
_output_shapes
: *
dtype0
f
	iterationVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_name	iteration
_
iteration/Read/ReadVariableOpReadVariableOp	iteration*
_output_shapes
: *
dtype0	
`
biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namebias
Y
bias/Read/ReadVariableOpReadVariableOpbias*
_output_shapes
:*
dtype0
h
kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*
shared_namekernel
a
kernel/Read/ReadVariableOpReadVariableOpkernel*
_output_shapes

:*
dtype0
d
bias_1VarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namebias_1
]
bias_1/Read/ReadVariableOpReadVariableOpbias_1*
_output_shapes
:*
dtype0
p
kernel_1VarHandleOp*
_output_shapes
: *
dtype0*
shape:**
shared_name
kernel_1
i
kernel_1/Read/ReadVariableOpReadVariableOpkernel_1*"
_output_shapes
:**
dtype0

serving_default_OFFSOURCEPlaceholder*-
_output_shapes
:џџџџџџџџџ*
dtype0*"
shape:џџџџџџџџџ

serving_default_ONSOURCEPlaceholder*,
_output_shapes
:џџџџџџџџџ *
dtype0*!
shape:џџџџџџџџџ 

StatefulPartitionedCallStatefulPartitionedCallserving_default_OFFSOURCEserving_default_ONSOURCEkernel_1bias_1kernelbias*
Tin

2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*&
_read_only_resource_inputs
*<
config_proto,*

CPU

GPU(2*0J

  zE8 *-
f(R&
$__inference_signature_wrapper_283708

NoOpNoOp
>
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*М=
valueВ=BЏ= BЈ=
Ї
layer-0
layer-1
layer-2
layer-3
layer-4
layer_with_weights-0
layer-5
layer-6
layer-7
	layer-8

layer_with_weights-1

layer-9
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
_default_save_signature
	optimizer

signatures
#_self_saveable_object_factories*
'
#_self_saveable_object_factories* 
Г
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
#_self_saveable_object_factories* 
Г
	variables
trainable_variables
regularization_losses
 	keras_api
!__call__
*"&call_and_return_all_conditional_losses
##_self_saveable_object_factories* 
Г
$	variables
%trainable_variables
&regularization_losses
'	keras_api
(__call__
*)&call_and_return_all_conditional_losses
#*_self_saveable_object_factories* 
Г
+	variables
,trainable_variables
-regularization_losses
.	keras_api
/__call__
*0&call_and_return_all_conditional_losses
#1_self_saveable_object_factories* 
э
2	variables
3trainable_variables
4regularization_losses
5	keras_api
6__call__
*7&call_and_return_all_conditional_losses

8kernel
9bias
#:_self_saveable_object_factories
 ;_jit_compiled_convolution_op*
Г
<	variables
=trainable_variables
>regularization_losses
?	keras_api
@__call__
*A&call_and_return_all_conditional_losses
#B_self_saveable_object_factories* 
Г
C	variables
Dtrainable_variables
Eregularization_losses
F	keras_api
G__call__
*H&call_and_return_all_conditional_losses
#I_self_saveable_object_factories* 
'
#J_self_saveable_object_factories* 
Ы
K	variables
Ltrainable_variables
Mregularization_losses
N	keras_api
O__call__
*P&call_and_return_all_conditional_losses

Qkernel
Rbias
#S_self_saveable_object_factories*
 
80
91
Q2
R3*
 
80
91
Q2
R3*
* 
А
Tnon_trainable_variables

Ulayers
Vmetrics
Wlayer_regularization_losses
Xlayer_metrics
	variables
trainable_variables
regularization_losses
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*
6
Ytrace_0
Ztrace_1
[trace_2
\trace_3* 
6
]trace_0
^trace_1
_trace_2
`trace_3* 
* 

a
_variables
b_iterations
c_learning_rate
d_index_dict
e
_momentums
f_velocities
g_update_step_xla*

hserving_default* 
* 
* 
* 
* 
* 

inon_trainable_variables

jlayers
kmetrics
llayer_regularization_losses
mlayer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses* 

ntrace_0* 

otrace_0* 
* 
* 
* 
* 

pnon_trainable_variables

qlayers
rmetrics
slayer_regularization_losses
tlayer_metrics
	variables
trainable_variables
regularization_losses
!__call__
*"&call_and_return_all_conditional_losses
&""call_and_return_conditional_losses* 

utrace_0* 

vtrace_0* 
* 
* 
* 
* 

wnon_trainable_variables

xlayers
ymetrics
zlayer_regularization_losses
{layer_metrics
$	variables
%trainable_variables
&regularization_losses
(__call__
*)&call_and_return_all_conditional_losses
&)"call_and_return_conditional_losses* 

|trace_0* 

}trace_0* 
* 
* 
* 
* 

~non_trainable_variables

layers
metrics
 layer_regularization_losses
layer_metrics
+	variables
,trainable_variables
-regularization_losses
/__call__
*0&call_and_return_all_conditional_losses
&0"call_and_return_conditional_losses* 

trace_0* 

trace_0* 
* 

80
91*

80
91*
* 

non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
2	variables
3trainable_variables
4regularization_losses
6__call__
*7&call_and_return_all_conditional_losses
&7"call_and_return_conditional_losses*

trace_0* 

trace_0* 
XR
VARIABLE_VALUEkernel_16layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE*
TN
VARIABLE_VALUEbias_14layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
* 
* 

non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
<	variables
=trainable_variables
>regularization_losses
@__call__
*A&call_and_return_all_conditional_losses
&A"call_and_return_conditional_losses* 

trace_0* 

trace_0* 
* 
* 
* 
* 

non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
C	variables
Dtrainable_variables
Eregularization_losses
G__call__
*H&call_and_return_all_conditional_losses
&H"call_and_return_conditional_losses* 

trace_0* 

trace_0* 
* 
* 

Q0
R1*

Q0
R1*
* 

non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
K	variables
Ltrainable_variables
Mregularization_losses
O__call__
*P&call_and_return_all_conditional_losses
&P"call_and_return_conditional_losses*

trace_0* 

 trace_0* 
VP
VARIABLE_VALUEkernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE*
RL
VARIABLE_VALUEbias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
J
0
1
2
3
4
5
6
7
	8

9*

Ё0
Ђ1*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
K
b0
Ѓ1
Є2
Ѕ3
І4
Ї5
Ј6
Љ7
Њ8*
SM
VARIABLE_VALUE	iteration0optimizer/_iterations/.ATTRIBUTES/VARIABLE_VALUE*
ZT
VARIABLE_VALUElearning_rate3optimizer/_learning_rate/.ATTRIBUTES/VARIABLE_VALUE*
* 
$
Ѓ0
Ѕ1
Ї2
Љ3*
$
Є0
І1
Ј2
Њ3*
:
Ћtrace_0
Ќtrace_1
­trace_2
Ўtrace_3* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
<
Џ	variables
А	keras_api

Бtotal

Вcount*
M
Г	variables
Д	keras_api

Еtotal

Жcount
З
_fn_kwargs*
ZT
VARIABLE_VALUEAdam/m/kernel_11optimizer/_variables/1/.ATTRIBUTES/VARIABLE_VALUE*
ZT
VARIABLE_VALUEAdam/v/kernel_11optimizer/_variables/2/.ATTRIBUTES/VARIABLE_VALUE*
XR
VARIABLE_VALUEAdam/m/bias_11optimizer/_variables/3/.ATTRIBUTES/VARIABLE_VALUE*
XR
VARIABLE_VALUEAdam/v/bias_11optimizer/_variables/4/.ATTRIBUTES/VARIABLE_VALUE*
XR
VARIABLE_VALUEAdam/m/kernel1optimizer/_variables/5/.ATTRIBUTES/VARIABLE_VALUE*
XR
VARIABLE_VALUEAdam/v/kernel1optimizer/_variables/6/.ATTRIBUTES/VARIABLE_VALUE*
VP
VARIABLE_VALUEAdam/m/bias1optimizer/_variables/7/.ATTRIBUTES/VARIABLE_VALUE*
VP
VARIABLE_VALUEAdam/v/bias1optimizer/_variables/8/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
* 

Б0
В1*

Џ	variables*
UO
VARIABLE_VALUEtotal_14keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE*
UO
VARIABLE_VALUEcount_14keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE*

Е0
Ж1*

Г	variables*
SM
VARIABLE_VALUEtotal4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUE*
SM
VARIABLE_VALUEcount4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUE*
* 
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 

StatefulPartitionedCall_1StatefulPartitionedCallsaver_filenamekernel_1bias_1kernelbias	iterationlearning_rateAdam/m/kernel_1Adam/v/kernel_1Adam/m/bias_1Adam/v/bias_1Adam/m/kernelAdam/v/kernelAdam/m/biasAdam/v/biastotal_1count_1totalcountConst*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *<
config_proto,*

CPU

GPU(2*0J

  zE8 *(
f#R!
__inference__traced_save_284227

StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamekernel_1bias_1kernelbias	iterationlearning_rateAdam/m/kernel_1Adam/v/kernel_1Adam/m/bias_1Adam/v/bias_1Adam/m/kernelAdam/v/kernelAdam/m/biasAdam/v/biastotal_1count_1totalcount*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *<
config_proto,*

CPU

GPU(2*0J

  zE8 *+
f&R$
"__inference__traced_restore_284291щ
О
b
F__inference_flatten_57_layer_call_and_return_conditional_losses_283524

inputs
identityV
ConstConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   \
ReshapeReshapeinputsConst:output:0*
T0*'
_output_shapes
:џџџџџџџџџX
IdentityIdentityReshape:output:0*
T0*'
_output_shapes
:џџџџџџџџџ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:џџџџџџџџџ:S O
+
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
а
h
L__inference_max_pooling1d_71_layer_call_and_return_conditional_losses_283472

inputs
identityP
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :

ExpandDims
ExpandDimsinputsExpandDims/dim:output:0*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџЅ
MaxPoolMaxPoolExpandDims:output:0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ*
ksize
*
paddingSAME*
strides

SqueezeSqueezeMaxPool:output:0*
T0*=
_output_shapes+
):'џџџџџџџџџџџџџџџџџџџџџџџџџџџ*
squeeze_dims
n
IdentityIdentitySqueeze:output:0*
T0*=
_output_shapes+
):'џџџџџџџџџџџџџџџџџџџџџџџџџџџ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):'џџџџџџџџџџџџџџџџџџџџџџџџџџџ:e a
=
_output_shapes+
):'џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
И
G
+__inference_flatten_57_layer_call_fn_283929

inputs
identityР
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *<
config_proto,*

CPU

GPU(2*0J

  zE8 *O
fJRH
F__inference_flatten_57_layer_call_and_return_conditional_losses_283524`
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:џџџџџџџџџ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:џџџџџџџџџ:S O
+
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs

M
1__inference_max_pooling1d_69_layer_call_fn_283857

inputs
identityм
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *=
_output_shapes+
):'џџџџџџџџџџџџџџџџџџџџџџџџџџџ* 
_read_only_resource_inputs
 *<
config_proto,*

CPU

GPU(2*0J

  zE8 *U
fPRN
L__inference_max_pooling1d_69_layer_call_and_return_conditional_losses_283442v
IdentityIdentityPartitionedCall:output:0*
T0*=
_output_shapes+
):'џџџџџџџџџџџџџџџџџџџџџџџџџџџ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):'џџџџџџџџџџџџџџџџџџџџџџџџџџџ:e a
=
_output_shapes+
):'џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
р
В
#__inference_internal_grad_fn_284079
result_grads_0
result_grads_1
result_grads_2
mul_conv1d_53_beta
mul_conv1d_53_biasadd
identity

identity_1|
mulMulmul_conv1d_53_betamul_conv1d_53_biasadd^result_grads_0*
T0*+
_output_shapes
:џџџџџџџџџQ
SigmoidSigmoidmul:z:0*
T0*+
_output_shapes
:џџџџџџџџџm
mul_1Mulmul_conv1d_53_betamul_conv1d_53_biasadd*
T0*+
_output_shapes
:џџџџџџџџџJ
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?]
subSubsub/x:output:0Sigmoid:y:0*
T0*+
_output_shapes
:џџџџџџџџџV
mul_2Mul	mul_1:z:0sub:z:0*
T0*+
_output_shapes
:џџџџџџџџџJ
add/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?]
addAddV2add/x:output:0	mul_2:z:0*
T0*+
_output_shapes
:џџџџџџџџџX
mul_3MulSigmoid:y:0add:z:0*
T0*+
_output_shapes
:џџџџџџџџџ]
SquareSquaremul_conv1d_53_biasadd*
T0*+
_output_shapes
:џџџџџџџџџ^
mul_4Mulresult_grads_0
Square:y:0*
T0*+
_output_shapes
:џџџџџџџџџZ
mul_5Mul	mul_4:z:0Sigmoid:y:0*
T0*+
_output_shapes
:џџџџџџџџџL
sub_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?a
sub_1Subsub_1/x:output:0Sigmoid:y:0*
T0*+
_output_shapes
:џџџџџџџџџX
mul_6Mul	mul_5:z:0	sub_1:z:0*
T0*+
_output_shapes
:џџџџџџџџџZ
ConstConst*
_output_shapes
:*
dtype0*!
valueB"          F
SumSum	mul_6:z:0Const:output:0*
T0*
_output_shapes
: ]
mul_7Mulresult_grads_0	mul_3:z:0*
T0*+
_output_shapes
:џџџџџџџџџU
IdentityIdentity	mul_7:z:0*
T0*+
_output_shapes
:џџџџџџџџџE

Identity_1IdentitySum:output:0*
T0*
_output_shapes
: "!

identity_1Identity_1:output:0"
identityIdentity:output:0*\
_input_shapesK
I:џџџџџџџџџ:џџџџџџџџџ: : :џџџџџџџџџ:1-
+
_output_shapes
:џџџџџџџџџ:

_output_shapes
: :FB

_output_shapes
: 
(
_user_specified_nameresult_grads_2:[W
+
_output_shapes
:џџџџџџџџџ
(
_user_specified_nameresult_grads_1: 
&
 _has_manual_control_dependencies(
+
_output_shapes
:џџџџџџџџџ
(
_user_specified_nameresult_grads_0
г 
к
D__inference_model_57_layer_call_and_return_conditional_losses_283565
	offsource
onsource&
conv1d_53_283552:*
conv1d_53_283554:(
injection_masks_283559:$
injection_masks_283561:
identityЂ'INJECTION_MASKS/StatefulPartitionedCallЂ!conv1d_53/StatefulPartitionedCallС
%whiten_passthrough_28/PartitionedCallPartitionedCall	offsource*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *<
config_proto,*

CPU

GPU(2*0J

  zE8 *2
f-R+
)__inference_restored_function_body_283383л
reshape_57/PartitionedCallPartitionedCall.whiten_passthrough_28/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *<
config_proto,*

CPU

GPU(2*0J

  zE8 *2
f-R+
)__inference_restored_function_body_283389ј
 max_pooling1d_69/PartitionedCallPartitionedCall#reshape_57/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:џџџџџџџџџb* 
_read_only_resource_inputs
 *<
config_proto,*

CPU

GPU(2*0J

  zE8 *U
fPRN
L__inference_max_pooling1d_69_layer_call_and_return_conditional_losses_283442ў
 max_pooling1d_70/PartitionedCallPartitionedCall)max_pooling1d_69/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *<
config_proto,*

CPU

GPU(2*0J

  zE8 *U
fPRN
L__inference_max_pooling1d_70_layer_call_and_return_conditional_losses_283457Њ
!conv1d_53/StatefulPartitionedCallStatefulPartitionedCall)max_pooling1d_70/PartitionedCall:output:0conv1d_53_283552conv1d_53_283554*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*<
config_proto,*

CPU

GPU(2*0J

  zE8 *N
fIRG
E__inference_conv1d_53_layer_call_and_return_conditional_losses_283511џ
 max_pooling1d_71/PartitionedCallPartitionedCall*conv1d_53/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *<
config_proto,*

CPU

GPU(2*0J

  zE8 *U
fPRN
L__inference_max_pooling1d_71_layer_call_and_return_conditional_losses_283472ю
flatten_57/PartitionedCallPartitionedCall)max_pooling1d_71/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *<
config_proto,*

CPU

GPU(2*0J

  zE8 *O
fJRH
F__inference_flatten_57_layer_call_and_return_conditional_losses_283524И
'INJECTION_MASKS/StatefulPartitionedCallStatefulPartitionedCall#flatten_57/PartitionedCall:output:0injection_masks_283559injection_masks_283561*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*<
config_proto,*

CPU

GPU(2*0J

  zE8 *T
fORM
K__inference_INJECTION_MASKS_layer_call_and_return_conditional_losses_283537
IdentityIdentity0INJECTION_MASKS/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ
NoOpNoOp(^INJECTION_MASKS/StatefulPartitionedCall"^conv1d_53/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*L
_input_shapes;
9:џџџџџџџџџ:џџџџџџџџџ : : : : 2R
'INJECTION_MASKS/StatefulPartitionedCall'INJECTION_MASKS/StatefulPartitionedCall2F
!conv1d_53/StatefulPartitionedCall!conv1d_53/StatefulPartitionedCall:VR
,
_output_shapes
:џџџџџџџџџ 
"
_user_specified_name
ONSOURCE:X T
-
_output_shapes
:џџџџџџџџџ
#
_user_specified_name	OFFSOURCE
­
E
)__inference_restored_function_body_283383

inputs
identityЎ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*,
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *<
config_proto,*

CPU

GPU(2*0J

  zE8 *W
fRRP
N__inference_whiten_passthrough_28_layer_call_and_return_conditional_losses_767e
IdentityIdentityPartitionedCall:output:0*
T0*,
_output_shapes
:џџџџџџџџџ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*,
_input_shapes
:џџџџџџџџџ:U Q
-
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
р
В
#__inference_internal_grad_fn_284107
result_grads_0
result_grads_1
result_grads_2
mul_conv1d_53_beta
mul_conv1d_53_biasadd
identity

identity_1|
mulMulmul_conv1d_53_betamul_conv1d_53_biasadd^result_grads_0*
T0*+
_output_shapes
:џџџџџџџџџQ
SigmoidSigmoidmul:z:0*
T0*+
_output_shapes
:џџџџџџџџџm
mul_1Mulmul_conv1d_53_betamul_conv1d_53_biasadd*
T0*+
_output_shapes
:џџџџџџџџџJ
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?]
subSubsub/x:output:0Sigmoid:y:0*
T0*+
_output_shapes
:џџџџџџџџџV
mul_2Mul	mul_1:z:0sub:z:0*
T0*+
_output_shapes
:џџџџџџџџџJ
add/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?]
addAddV2add/x:output:0	mul_2:z:0*
T0*+
_output_shapes
:џџџџџџџџџX
mul_3MulSigmoid:y:0add:z:0*
T0*+
_output_shapes
:џџџџџџџџџ]
SquareSquaremul_conv1d_53_biasadd*
T0*+
_output_shapes
:џџџџџџџџџ^
mul_4Mulresult_grads_0
Square:y:0*
T0*+
_output_shapes
:џџџџџџџџџZ
mul_5Mul	mul_4:z:0Sigmoid:y:0*
T0*+
_output_shapes
:џџџџџџџџџL
sub_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?a
sub_1Subsub_1/x:output:0Sigmoid:y:0*
T0*+
_output_shapes
:џџџџџџџџџX
mul_6Mul	mul_5:z:0	sub_1:z:0*
T0*+
_output_shapes
:џџџџџџџџџZ
ConstConst*
_output_shapes
:*
dtype0*!
valueB"          F
SumSum	mul_6:z:0Const:output:0*
T0*
_output_shapes
: ]
mul_7Mulresult_grads_0	mul_3:z:0*
T0*+
_output_shapes
:џџџџџџџџџU
IdentityIdentity	mul_7:z:0*
T0*+
_output_shapes
:џџџџџџџџџE

Identity_1IdentitySum:output:0*
T0*
_output_shapes
: "!

identity_1Identity_1:output:0"
identityIdentity:output:0*\
_input_shapesK
I:џџџџџџџџџ:џџџџџџџџџ: : :џџџџџџџџџ:1-
+
_output_shapes
:џџџџџџџџџ:

_output_shapes
: :FB

_output_shapes
: 
(
_user_specified_nameresult_grads_2:[W
+
_output_shapes
:џџџџџџџџџ
(
_user_specified_nameresult_grads_1: 
&
 _has_manual_control_dependencies(
+
_output_shapes
:џџџџџџџџџ
(
_user_specified_nameresult_grads_0
Ш
B
__inference_crop_samples_663
batched_onsource
identityd
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"     <  f
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"     D  f
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ц
strided_sliceStridedSlicebatched_onsourcestrided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*,
_output_shapes
:џџџџџџџџџ*
ellipsis_maskc
IdentityIdentitystrided_slice:output:0*
T0*,
_output_shapes
:џџџџџџџџџ"
identityIdentity:output:0*
_XlaMustCompile(*(
_construction_contextkEagerRuntime*,
_input_shapes
:џџџџџџџџџ*
	_noinline(:_ [
-
_output_shapes
:џџџџџџџџџ
*
_user_specified_namebatched_onsource
ё	
я
)__inference_model_57_layer_call_fn_283736
inputs_offsource
inputs_onsource
unknown:*
	unknown_0:
	unknown_1:
	unknown_2:
identityЂStatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputs_offsourceinputs_onsourceunknown	unknown_0	unknown_1	unknown_2*
Tin

2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*&
_read_only_resource_inputs
*<
config_proto,*

CPU

GPU(2*0J

  zE8 *M
fHRF
D__inference_model_57_layer_call_and_return_conditional_losses_283625o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*L
_input_shapes;
9:џџџџџџџџџ:џџџџџџџџџ : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:]Y
,
_output_shapes
:џџџџџџџџџ 
)
_user_specified_nameinputs_onsource:_ [
-
_output_shapes
:џџџџџџџџџ
*
_user_specified_nameinputs_offsource
г 
к
D__inference_model_57_layer_call_and_return_conditional_losses_283544
	offsource
onsource&
conv1d_53_283512:*
conv1d_53_283514:(
injection_masks_283538:$
injection_masks_283540:
identityЂ'INJECTION_MASKS/StatefulPartitionedCallЂ!conv1d_53/StatefulPartitionedCallС
%whiten_passthrough_28/PartitionedCallPartitionedCall	offsource*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *<
config_proto,*

CPU

GPU(2*0J

  zE8 *2
f-R+
)__inference_restored_function_body_283383л
reshape_57/PartitionedCallPartitionedCall.whiten_passthrough_28/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *<
config_proto,*

CPU

GPU(2*0J

  zE8 *2
f-R+
)__inference_restored_function_body_283389ј
 max_pooling1d_69/PartitionedCallPartitionedCall#reshape_57/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:џџџџџџџџџb* 
_read_only_resource_inputs
 *<
config_proto,*

CPU

GPU(2*0J

  zE8 *U
fPRN
L__inference_max_pooling1d_69_layer_call_and_return_conditional_losses_283442ў
 max_pooling1d_70/PartitionedCallPartitionedCall)max_pooling1d_69/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *<
config_proto,*

CPU

GPU(2*0J

  zE8 *U
fPRN
L__inference_max_pooling1d_70_layer_call_and_return_conditional_losses_283457Њ
!conv1d_53/StatefulPartitionedCallStatefulPartitionedCall)max_pooling1d_70/PartitionedCall:output:0conv1d_53_283512conv1d_53_283514*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*<
config_proto,*

CPU

GPU(2*0J

  zE8 *N
fIRG
E__inference_conv1d_53_layer_call_and_return_conditional_losses_283511џ
 max_pooling1d_71/PartitionedCallPartitionedCall*conv1d_53/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *<
config_proto,*

CPU

GPU(2*0J

  zE8 *U
fPRN
L__inference_max_pooling1d_71_layer_call_and_return_conditional_losses_283472ю
flatten_57/PartitionedCallPartitionedCall)max_pooling1d_71/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *<
config_proto,*

CPU

GPU(2*0J

  zE8 *O
fJRH
F__inference_flatten_57_layer_call_and_return_conditional_losses_283524И
'INJECTION_MASKS/StatefulPartitionedCallStatefulPartitionedCall#flatten_57/PartitionedCall:output:0injection_masks_283538injection_masks_283540*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*<
config_proto,*

CPU

GPU(2*0J

  zE8 *T
fORM
K__inference_INJECTION_MASKS_layer_call_and_return_conditional_losses_283537
IdentityIdentity0INJECTION_MASKS/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ
NoOpNoOp(^INJECTION_MASKS/StatefulPartitionedCall"^conv1d_53/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*L
_input_shapes;
9:џџџџџџџџџ:џџџџџџџџџ : : : : 2R
'INJECTION_MASKS/StatefulPartitionedCall'INJECTION_MASKS/StatefulPartitionedCall2F
!conv1d_53/StatefulPartitionedCall!conv1d_53/StatefulPartitionedCall:VR
,
_output_shapes
:џџџџџџџџџ 
"
_user_specified_name
ONSOURCE:X T
-
_output_shapes
:џџџџџџџџџ
#
_user_specified_name	OFFSOURCE
о
_
C__inference_reshape_57_layer_call_and_return_conditional_losses_512

inputs
identityc
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          n
	transpose	Transposeinputstranspose/perm:output:0*
T0*,
_output_shapes
:џџџџџџџџџZ
IdentityIdentitytranspose:y:0*
T0*,
_output_shapes
:џџџџџџџџџ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:џџџџџџџџџ:T P
,
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
ё	
я
)__inference_model_57_layer_call_fn_283722
inputs_offsource
inputs_onsource
unknown:*
	unknown_0:
	unknown_1:
	unknown_2:
identityЂStatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputs_offsourceinputs_onsourceunknown	unknown_0	unknown_1	unknown_2*
Tin

2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*&
_read_only_resource_inputs
*<
config_proto,*

CPU

GPU(2*0J

  zE8 *M
fHRF
D__inference_model_57_layer_call_and_return_conditional_losses_283590o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*L
_input_shapes;
9:џџџџџџџџџ:џџџџџџџџџ : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:]Y
,
_output_shapes
:џџџџџџџџџ 
)
_user_specified_nameinputs_onsource:_ [
-
_output_shapes
:џџџџџџџџџ
*
_user_specified_nameinputs_offsource


#__inference_internal_grad_fn_284023
result_grads_0
result_grads_1
result_grads_2
mul_beta
mul_biasadd
identity

identity_1h
mulMulmul_betamul_biasadd^result_grads_0*
T0*+
_output_shapes
:џџџџџџџџџQ
SigmoidSigmoidmul:z:0*
T0*+
_output_shapes
:џџџџџџџџџY
mul_1Mulmul_betamul_biasadd*
T0*+
_output_shapes
:џџџџџџџџџJ
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?]
subSubsub/x:output:0Sigmoid:y:0*
T0*+
_output_shapes
:џџџџџџџџџV
mul_2Mul	mul_1:z:0sub:z:0*
T0*+
_output_shapes
:џџџџџџџџџJ
add/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?]
addAddV2add/x:output:0	mul_2:z:0*
T0*+
_output_shapes
:џџџџџџџџџX
mul_3MulSigmoid:y:0add:z:0*
T0*+
_output_shapes
:џџџџџџџџџS
SquareSquaremul_biasadd*
T0*+
_output_shapes
:џџџџџџџџџ^
mul_4Mulresult_grads_0
Square:y:0*
T0*+
_output_shapes
:џџџџџџџџџZ
mul_5Mul	mul_4:z:0Sigmoid:y:0*
T0*+
_output_shapes
:џџџџџџџџџL
sub_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?a
sub_1Subsub_1/x:output:0Sigmoid:y:0*
T0*+
_output_shapes
:џџџџџџџџџX
mul_6Mul	mul_5:z:0	sub_1:z:0*
T0*+
_output_shapes
:џџџџџџџџџZ
ConstConst*
_output_shapes
:*
dtype0*!
valueB"          F
SumSum	mul_6:z:0Const:output:0*
T0*
_output_shapes
: ]
mul_7Mulresult_grads_0	mul_3:z:0*
T0*+
_output_shapes
:џџџџџџџџџU
IdentityIdentity	mul_7:z:0*
T0*+
_output_shapes
:џџџџџџџџџE

Identity_1IdentitySum:output:0*
T0*
_output_shapes
: "!

identity_1Identity_1:output:0"
identityIdentity:output:0*\
_input_shapesK
I:џџџџџџџџџ:џџџџџџџџџ: : :џџџџџџџџџ:1-
+
_output_shapes
:џџџџџџџџџ:

_output_shapes
: :FB

_output_shapes
: 
(
_user_specified_nameresult_grads_2:[W
+
_output_shapes
:џџџџџџџџџ
(
_user_specified_nameresult_grads_1: 
&
 _has_manual_control_dependencies(
+
_output_shapes
:џџџџџџџџџ
(
_user_specified_nameresult_grads_0
M
џ	
"__inference__traced_restore_284291
file_prefix/
assignvariableop_kernel_1:*'
assignvariableop_1_bias_1:+
assignvariableop_2_kernel:%
assignvariableop_3_bias:&
assignvariableop_4_iteration:	 *
 assignvariableop_5_learning_rate: 8
"assignvariableop_6_adam_m_kernel_1:*8
"assignvariableop_7_adam_v_kernel_1:*.
 assignvariableop_8_adam_m_bias_1:.
 assignvariableop_9_adam_v_bias_1:3
!assignvariableop_10_adam_m_kernel:3
!assignvariableop_11_adam_v_kernel:-
assignvariableop_12_adam_m_bias:-
assignvariableop_13_adam_v_bias:%
assignvariableop_14_total_1: %
assignvariableop_15_count_1: #
assignvariableop_16_total: #
assignvariableop_17_count: 
identity_19ЂAssignVariableOpЂAssignVariableOp_1ЂAssignVariableOp_10ЂAssignVariableOp_11ЂAssignVariableOp_12ЂAssignVariableOp_13ЂAssignVariableOp_14ЂAssignVariableOp_15ЂAssignVariableOp_16ЂAssignVariableOp_17ЂAssignVariableOp_2ЂAssignVariableOp_3ЂAssignVariableOp_4ЂAssignVariableOp_5ЂAssignVariableOp_6ЂAssignVariableOp_7ЂAssignVariableOp_8ЂAssignVariableOp_9Р
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*ц
valueмBйB6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB0optimizer/_iterations/.ATTRIBUTES/VARIABLE_VALUEB3optimizer/_learning_rate/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/1/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/2/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/3/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/4/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/5/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/6/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/7/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/8/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*9
value0B.B B B B B B B B B B B B B B B B B B B §
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*`
_output_shapesN
L:::::::::::::::::::*!
dtypes
2	[
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:Ќ
AssignVariableOpAssignVariableOpassignvariableop_kernel_1Identity:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:А
AssignVariableOp_1AssignVariableOpassignvariableop_1_bias_1Identity_1:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:А
AssignVariableOp_2AssignVariableOpassignvariableop_2_kernelIdentity_2:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:Ў
AssignVariableOp_3AssignVariableOpassignvariableop_3_biasIdentity_3:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0	*
_output_shapes
:Г
AssignVariableOp_4AssignVariableOpassignvariableop_4_iterationIdentity_4:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0	]

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:З
AssignVariableOp_5AssignVariableOp assignvariableop_5_learning_rateIdentity_5:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:Й
AssignVariableOp_6AssignVariableOp"assignvariableop_6_adam_m_kernel_1Identity_6:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:Й
AssignVariableOp_7AssignVariableOp"assignvariableop_7_adam_v_kernel_1Identity_7:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:З
AssignVariableOp_8AssignVariableOp assignvariableop_8_adam_m_bias_1Identity_8:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:З
AssignVariableOp_9AssignVariableOp assignvariableop_9_adam_v_bias_1Identity_9:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:К
AssignVariableOp_10AssignVariableOp!assignvariableop_10_adam_m_kernelIdentity_10:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:К
AssignVariableOp_11AssignVariableOp!assignvariableop_11_adam_v_kernelIdentity_11:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:И
AssignVariableOp_12AssignVariableOpassignvariableop_12_adam_m_biasIdentity_12:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:И
AssignVariableOp_13AssignVariableOpassignvariableop_13_adam_v_biasIdentity_13:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:Д
AssignVariableOp_14AssignVariableOpassignvariableop_14_total_1Identity_14:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:Д
AssignVariableOp_15AssignVariableOpassignvariableop_15_count_1Identity_15:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:В
AssignVariableOp_16AssignVariableOpassignvariableop_16_totalIdentity_16:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:В
AssignVariableOp_17AssignVariableOpassignvariableop_17_countIdentity_17:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0Y
NoOpNoOp"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 л
Identity_18Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: W
Identity_19IdentityIdentity_18:output:0^NoOp_1*
T0*
_output_shapes
: Ш
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*"
_acd_function_control_output(*
_output_shapes
 "#
identity_19Identity_19:output:0*9
_input_shapes(
&: : : : : : : : : : : : : : : : : : : 2*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_16AssignVariableOp_162*
AssignVariableOp_17AssignVariableOp_172(
AssignVariableOp_1AssignVariableOp_12(
AssignVariableOp_2AssignVariableOp_22(
AssignVariableOp_3AssignVariableOp_32(
AssignVariableOp_4AssignVariableOp_42(
AssignVariableOp_5AssignVariableOp_52(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_92$
AssignVariableOpAssignVariableOp:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
Ч	
с
)__inference_model_57_layer_call_fn_283636
	offsource
onsource
unknown:*
	unknown_0:
	unknown_1:
	unknown_2:
identityЂStatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCall	offsourceonsourceunknown	unknown_0	unknown_1	unknown_2*
Tin

2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*&
_read_only_resource_inputs
*<
config_proto,*

CPU

GPU(2*0J

  zE8 *M
fHRF
D__inference_model_57_layer_call_and_return_conditional_losses_283625o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*L
_input_shapes;
9:џџџџџџџџџ:џџџџџџџџџ : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:VR
,
_output_shapes
:џџџџџџџџџ 
"
_user_specified_name
ONSOURCE:X T
-
_output_shapes
:џџџџџџџџџ
#
_user_specified_name	OFFSOURCE
а
h
L__inference_max_pooling1d_69_layer_call_and_return_conditional_losses_283442

inputs
identityP
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :

ExpandDims
ExpandDimsinputsExpandDims/dim:output:0*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџЅ
MaxPoolMaxPoolExpandDims:output:0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ*
ksize
*
paddingSAME*
strides

SqueezeSqueezeMaxPool:output:0*
T0*=
_output_shapes+
):'џџџџџџџџџџџџџџџџџџџџџџџџџџџ*
squeeze_dims
n
IdentityIdentitySqueeze:output:0*
T0*=
_output_shapes+
):'џџџџџџџџџџџџџџџџџџџџџџџџџџџ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):'џџџџџџџџџџџџџџџџџџџџџџџџџџџ:e a
=
_output_shapes+
):'џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
а
h
L__inference_max_pooling1d_69_layer_call_and_return_conditional_losses_283865

inputs
identityP
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :

ExpandDims
ExpandDimsinputsExpandDims/dim:output:0*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџЅ
MaxPoolMaxPoolExpandDims:output:0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ*
ksize
*
paddingSAME*
strides

SqueezeSqueezeMaxPool:output:0*
T0*=
_output_shapes+
):'џџџџџџџџџџџџџџџџџџџџџџџџџџџ*
squeeze_dims
n
IdentityIdentitySqueeze:output:0*
T0*=
_output_shapes+
):'џџџџџџџџџџџџџџџџџџџџџџџџџџџ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):'џџџџџџџџџџџџџџџџџџџџџџџџџџџ:e a
=
_output_shapes+
):'џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
а
h
L__inference_max_pooling1d_70_layer_call_and_return_conditional_losses_283457

inputs
identityP
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :

ExpandDims
ExpandDimsinputsExpandDims/dim:output:0*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџЅ
MaxPoolMaxPoolExpandDims:output:0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ*
ksize
*
paddingSAME*
strides

SqueezeSqueezeMaxPool:output:0*
T0*=
_output_shapes+
):'џџџџџџџџџџџџџџџџџџџџџџџџџџџ*
squeeze_dims
n
IdentityIdentitySqueeze:output:0*
T0*=
_output_shapes+
):'џџџџџџџџџџџџџџџџџџџџџџџџџџџ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):'џџџџџџџџџџџџџџџџџџџџџџџџџџџ:e a
=
_output_shapes+
):'џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
ќ

E__inference_conv1d_53_layer_call_and_return_conditional_losses_283511

inputsA
+conv1d_expanddims_1_readvariableop_resource:*-
biasadd_readvariableop_resource:

identity_1ЂBiasAdd/ReadVariableOpЂ"Conv1D/ExpandDims_1/ReadVariableOp`
Conv1D/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
§џџџџџџџџ
Conv1D/ExpandDims
ExpandDimsinputsConv1D/ExpandDims/dim:output:0*
T0*/
_output_shapes
:џџџџџџџџџ
"Conv1D/ExpandDims_1/ReadVariableOpReadVariableOp+conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:**
dtype0Y
Conv1D/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B :  
Conv1D/ExpandDims_1
ExpandDims*Conv1D/ExpandDims_1/ReadVariableOp:value:0 Conv1D/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:*Ќ
Conv1DConv2DConv1D/ExpandDims:output:0Conv1D/ExpandDims_1:output:0*
T0*/
_output_shapes
:џџџџџџџџџ*
paddingSAME*
strides
)
Conv1D/SqueezeSqueezeConv1D:output:0*
T0*+
_output_shapes
:џџџџџџџџџ*
squeeze_dims

§џџџџџџџџr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0
BiasAddBiasAddConv1D/Squeeze:output:0BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:џџџџџџџџџI
betaConst*
_output_shapes
: *
dtype0*
valueB
 *  ?a
mulMulbeta:output:0BiasAdd:output:0*
T0*+
_output_shapes
:џџџџџџџџџQ
SigmoidSigmoidmul:z:0*
T0*+
_output_shapes
:џџџџџџџџџa
mul_1MulBiasAdd:output:0Sigmoid:y:0*
T0*+
_output_shapes
:џџџџџџџџџU
IdentityIdentity	mul_1:z:0*
T0*+
_output_shapes
:џџџџџџџџџФ
	IdentityN	IdentityN	mul_1:z:0BiasAdd:output:0beta:output:0*
T
2*,
_gradient_op_typeCustomGradient-283502*D
_output_shapes2
0:џџџџџџџџџ:џџџџџџџџџ: g

Identity_1IdentityIdentityN:output:0^NoOp*
T0*+
_output_shapes
:џџџџџџџџџ
NoOpNoOp^BiasAdd/ReadVariableOp#^Conv1D/ExpandDims_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:џџџџџџџџџ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2H
"Conv1D/ExpandDims_1/ReadVariableOp"Conv1D/ExpandDims_1/ReadVariableOp:S O
+
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
п

0__inference_INJECTION_MASKS_layer_call_fn_283944

inputs
unknown:
	unknown_0:
identityЂStatefulPartitionedCallя
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*<
config_proto,*

CPU

GPU(2*0J

  zE8 *T
fORM
K__inference_INJECTION_MASKS_layer_call_and_return_conditional_losses_283537o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:џџџџџџџџџ: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
ж
O
3__inference_whiten_passthrough_28_layer_call_fn_750

inputs
identityЭ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *<
config_proto,*

CPU

GPU(2*0J

 @E8 *W
fRRP
N__inference_whiten_passthrough_28_layer_call_and_return_conditional_losses_745e
IdentityIdentityPartitionedCall:output:0*
T0*,
_output_shapes
:џџџџџџџџџ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*,
_input_shapes
:џџџџџџџџџ:U Q
-
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
Ё

ќ
K__inference_INJECTION_MASKS_layer_call_and_return_conditional_losses_283537

inputs0
matmul_readvariableop_resource:-
biasadd_readvariableop_resource:
identityЂBiasAdd/ReadVariableOpЂMatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџV
SigmoidSigmoidBiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџZ
IdentityIdentitySigmoid:y:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџw
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:џџџџџџџџџ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
ќ

E__inference_conv1d_53_layer_call_and_return_conditional_losses_283911

inputsA
+conv1d_expanddims_1_readvariableop_resource:*-
biasadd_readvariableop_resource:

identity_1ЂBiasAdd/ReadVariableOpЂ"Conv1D/ExpandDims_1/ReadVariableOp`
Conv1D/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
§џџџџџџџџ
Conv1D/ExpandDims
ExpandDimsinputsConv1D/ExpandDims/dim:output:0*
T0*/
_output_shapes
:џџџџџџџџџ
"Conv1D/ExpandDims_1/ReadVariableOpReadVariableOp+conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:**
dtype0Y
Conv1D/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B :  
Conv1D/ExpandDims_1
ExpandDims*Conv1D/ExpandDims_1/ReadVariableOp:value:0 Conv1D/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:*Ќ
Conv1DConv2DConv1D/ExpandDims:output:0Conv1D/ExpandDims_1:output:0*
T0*/
_output_shapes
:џџџџџџџџџ*
paddingSAME*
strides
)
Conv1D/SqueezeSqueezeConv1D:output:0*
T0*+
_output_shapes
:џџџџџџџџџ*
squeeze_dims

§џџџџџџџџr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0
BiasAddBiasAddConv1D/Squeeze:output:0BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:џџџџџџџџџI
betaConst*
_output_shapes
: *
dtype0*
valueB
 *  ?a
mulMulbeta:output:0BiasAdd:output:0*
T0*+
_output_shapes
:џџџџџџџџџQ
SigmoidSigmoidmul:z:0*
T0*+
_output_shapes
:џџџџџџџџџa
mul_1MulBiasAdd:output:0Sigmoid:y:0*
T0*+
_output_shapes
:џџџџџџџџџU
IdentityIdentity	mul_1:z:0*
T0*+
_output_shapes
:џџџџџџџџџФ
	IdentityN	IdentityN	mul_1:z:0BiasAdd:output:0beta:output:0*
T
2*,
_gradient_op_typeCustomGradient-283902*D
_output_shapes2
0:џџџџџџџџџ:џџџџџџџџџ: g

Identity_1IdentityIdentityN:output:0^NoOp*
T0*+
_output_shapes
:џџџџџџџџџ
NoOpNoOp^BiasAdd/ReadVariableOp#^Conv1D/ExpandDims_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:џџџџџџџџџ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2H
"Conv1D/ExpandDims_1/ReadVariableOp"Conv1D/ExpandDims_1/ReadVariableOp:S O
+
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
Ќ
K
#__inference__update_step_xla_283852
gradient
variable:*
_XlaMustCompile(*(
_construction_contextkEagerRuntime*
_input_shapes

:: *
	_noinline(:($
"
_user_specified_name
variable:D @

_output_shapes
:
"
_user_specified_name
gradient
И
O
#__inference__update_step_xla_283847
gradient
variable:*
_XlaMustCompile(*(
_construction_contextkEagerRuntime*
_input_shapes
:: *
	_noinline(:($
"
_user_specified_name
variable:H D

_output_shapes

:
"
_user_specified_name
gradient
є
j
N__inference_whiten_passthrough_28_layer_call_and_return_conditional_losses_767

inputs
identityВ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_XlaMustCompile(*
_collective_manager_ids
 *,
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *<
config_proto,*

CPU

GPU(2*0J

 @E8 *%
f R
__inference_crop_samples_663I
ShapeShapeinputs*
T0*
_output_shapes
::эЯ]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:б
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:й
strided_slice_1StridedSliceShape:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskR
Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value
B :
Reshape/shapePackstrided_slice:output:0strided_slice_1:output:0Reshape/shape/2:output:0*
N*
T0*
_output_shapes
:{
ReshapeReshapePartitionedCall:output:0Reshape/shape:output:0*
T0*,
_output_shapes
:џџџџџџџџџ]
IdentityIdentityReshape:output:0*
T0*,
_output_shapes
:џџџџџџџџџ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*,
_input_shapes
:џџџџџџџџџ:U Q
-
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
Щ
Б
__inference__traced_save_284227
file_prefix5
read_disablecopyonread_kernel_1:*-
read_1_disablecopyonread_bias_1:1
read_2_disablecopyonread_kernel:+
read_3_disablecopyonread_bias:,
"read_4_disablecopyonread_iteration:	 0
&read_5_disablecopyonread_learning_rate: >
(read_6_disablecopyonread_adam_m_kernel_1:*>
(read_7_disablecopyonread_adam_v_kernel_1:*4
&read_8_disablecopyonread_adam_m_bias_1:4
&read_9_disablecopyonread_adam_v_bias_1:9
'read_10_disablecopyonread_adam_m_kernel:9
'read_11_disablecopyonread_adam_v_kernel:3
%read_12_disablecopyonread_adam_m_bias:3
%read_13_disablecopyonread_adam_v_bias:+
!read_14_disablecopyonread_total_1: +
!read_15_disablecopyonread_count_1: )
read_16_disablecopyonread_total: )
read_17_disablecopyonread_count: 
savev2_const
identity_37ЂMergeV2CheckpointsЂRead/DisableCopyOnReadЂRead/ReadVariableOpЂRead_1/DisableCopyOnReadЂRead_1/ReadVariableOpЂRead_10/DisableCopyOnReadЂRead_10/ReadVariableOpЂRead_11/DisableCopyOnReadЂRead_11/ReadVariableOpЂRead_12/DisableCopyOnReadЂRead_12/ReadVariableOpЂRead_13/DisableCopyOnReadЂRead_13/ReadVariableOpЂRead_14/DisableCopyOnReadЂRead_14/ReadVariableOpЂRead_15/DisableCopyOnReadЂRead_15/ReadVariableOpЂRead_16/DisableCopyOnReadЂRead_16/ReadVariableOpЂRead_17/DisableCopyOnReadЂRead_17/ReadVariableOpЂRead_2/DisableCopyOnReadЂRead_2/ReadVariableOpЂRead_3/DisableCopyOnReadЂRead_3/ReadVariableOpЂRead_4/DisableCopyOnReadЂRead_4/ReadVariableOpЂRead_5/DisableCopyOnReadЂRead_5/ReadVariableOpЂRead_6/DisableCopyOnReadЂRead_6/ReadVariableOpЂRead_7/DisableCopyOnReadЂRead_7/ReadVariableOpЂRead_8/DisableCopyOnReadЂRead_8/ReadVariableOpЂRead_9/DisableCopyOnReadЂRead_9/ReadVariableOpw
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*Z
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.parta
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: f

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: L

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :f
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : 
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: q
Read/DisableCopyOnReadDisableCopyOnReadread_disablecopyonread_kernel_1"/device:CPU:0*
_output_shapes
 
Read/ReadVariableOpReadVariableOpread_disablecopyonread_kernel_1^Read/DisableCopyOnRead"/device:CPU:0*"
_output_shapes
:**
dtype0m
IdentityIdentityRead/ReadVariableOp:value:0"/device:CPU:0*
T0*"
_output_shapes
:*e

Identity_1IdentityIdentity:output:0"/device:CPU:0*
T0*"
_output_shapes
:*s
Read_1/DisableCopyOnReadDisableCopyOnReadread_1_disablecopyonread_bias_1"/device:CPU:0*
_output_shapes
 
Read_1/ReadVariableOpReadVariableOpread_1_disablecopyonread_bias_1^Read_1/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0i

Identity_2IdentityRead_1/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:_

Identity_3IdentityIdentity_2:output:0"/device:CPU:0*
T0*
_output_shapes
:s
Read_2/DisableCopyOnReadDisableCopyOnReadread_2_disablecopyonread_kernel"/device:CPU:0*
_output_shapes
 
Read_2/ReadVariableOpReadVariableOpread_2_disablecopyonread_kernel^Read_2/DisableCopyOnRead"/device:CPU:0*
_output_shapes

:*
dtype0m

Identity_4IdentityRead_2/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

:c

Identity_5IdentityIdentity_4:output:0"/device:CPU:0*
T0*
_output_shapes

:q
Read_3/DisableCopyOnReadDisableCopyOnReadread_3_disablecopyonread_bias"/device:CPU:0*
_output_shapes
 
Read_3/ReadVariableOpReadVariableOpread_3_disablecopyonread_bias^Read_3/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0i

Identity_6IdentityRead_3/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:_

Identity_7IdentityIdentity_6:output:0"/device:CPU:0*
T0*
_output_shapes
:v
Read_4/DisableCopyOnReadDisableCopyOnRead"read_4_disablecopyonread_iteration"/device:CPU:0*
_output_shapes
 
Read_4/ReadVariableOpReadVariableOp"read_4_disablecopyonread_iteration^Read_4/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0	e

Identity_8IdentityRead_4/ReadVariableOp:value:0"/device:CPU:0*
T0	*
_output_shapes
: [

Identity_9IdentityIdentity_8:output:0"/device:CPU:0*
T0	*
_output_shapes
: z
Read_5/DisableCopyOnReadDisableCopyOnRead&read_5_disablecopyonread_learning_rate"/device:CPU:0*
_output_shapes
 
Read_5/ReadVariableOpReadVariableOp&read_5_disablecopyonread_learning_rate^Read_5/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0f
Identity_10IdentityRead_5/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: ]
Identity_11IdentityIdentity_10:output:0"/device:CPU:0*
T0*
_output_shapes
: |
Read_6/DisableCopyOnReadDisableCopyOnRead(read_6_disablecopyonread_adam_m_kernel_1"/device:CPU:0*
_output_shapes
 Ќ
Read_6/ReadVariableOpReadVariableOp(read_6_disablecopyonread_adam_m_kernel_1^Read_6/DisableCopyOnRead"/device:CPU:0*"
_output_shapes
:**
dtype0r
Identity_12IdentityRead_6/ReadVariableOp:value:0"/device:CPU:0*
T0*"
_output_shapes
:*i
Identity_13IdentityIdentity_12:output:0"/device:CPU:0*
T0*"
_output_shapes
:*|
Read_7/DisableCopyOnReadDisableCopyOnRead(read_7_disablecopyonread_adam_v_kernel_1"/device:CPU:0*
_output_shapes
 Ќ
Read_7/ReadVariableOpReadVariableOp(read_7_disablecopyonread_adam_v_kernel_1^Read_7/DisableCopyOnRead"/device:CPU:0*"
_output_shapes
:**
dtype0r
Identity_14IdentityRead_7/ReadVariableOp:value:0"/device:CPU:0*
T0*"
_output_shapes
:*i
Identity_15IdentityIdentity_14:output:0"/device:CPU:0*
T0*"
_output_shapes
:*z
Read_8/DisableCopyOnReadDisableCopyOnRead&read_8_disablecopyonread_adam_m_bias_1"/device:CPU:0*
_output_shapes
 Ђ
Read_8/ReadVariableOpReadVariableOp&read_8_disablecopyonread_adam_m_bias_1^Read_8/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0j
Identity_16IdentityRead_8/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:a
Identity_17IdentityIdentity_16:output:0"/device:CPU:0*
T0*
_output_shapes
:z
Read_9/DisableCopyOnReadDisableCopyOnRead&read_9_disablecopyonread_adam_v_bias_1"/device:CPU:0*
_output_shapes
 Ђ
Read_9/ReadVariableOpReadVariableOp&read_9_disablecopyonread_adam_v_bias_1^Read_9/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0j
Identity_18IdentityRead_9/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:a
Identity_19IdentityIdentity_18:output:0"/device:CPU:0*
T0*
_output_shapes
:|
Read_10/DisableCopyOnReadDisableCopyOnRead'read_10_disablecopyonread_adam_m_kernel"/device:CPU:0*
_output_shapes
 Љ
Read_10/ReadVariableOpReadVariableOp'read_10_disablecopyonread_adam_m_kernel^Read_10/DisableCopyOnRead"/device:CPU:0*
_output_shapes

:*
dtype0o
Identity_20IdentityRead_10/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

:e
Identity_21IdentityIdentity_20:output:0"/device:CPU:0*
T0*
_output_shapes

:|
Read_11/DisableCopyOnReadDisableCopyOnRead'read_11_disablecopyonread_adam_v_kernel"/device:CPU:0*
_output_shapes
 Љ
Read_11/ReadVariableOpReadVariableOp'read_11_disablecopyonread_adam_v_kernel^Read_11/DisableCopyOnRead"/device:CPU:0*
_output_shapes

:*
dtype0o
Identity_22IdentityRead_11/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

:e
Identity_23IdentityIdentity_22:output:0"/device:CPU:0*
T0*
_output_shapes

:z
Read_12/DisableCopyOnReadDisableCopyOnRead%read_12_disablecopyonread_adam_m_bias"/device:CPU:0*
_output_shapes
 Ѓ
Read_12/ReadVariableOpReadVariableOp%read_12_disablecopyonread_adam_m_bias^Read_12/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0k
Identity_24IdentityRead_12/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:a
Identity_25IdentityIdentity_24:output:0"/device:CPU:0*
T0*
_output_shapes
:z
Read_13/DisableCopyOnReadDisableCopyOnRead%read_13_disablecopyonread_adam_v_bias"/device:CPU:0*
_output_shapes
 Ѓ
Read_13/ReadVariableOpReadVariableOp%read_13_disablecopyonread_adam_v_bias^Read_13/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0k
Identity_26IdentityRead_13/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:a
Identity_27IdentityIdentity_26:output:0"/device:CPU:0*
T0*
_output_shapes
:v
Read_14/DisableCopyOnReadDisableCopyOnRead!read_14_disablecopyonread_total_1"/device:CPU:0*
_output_shapes
 
Read_14/ReadVariableOpReadVariableOp!read_14_disablecopyonread_total_1^Read_14/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0g
Identity_28IdentityRead_14/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: ]
Identity_29IdentityIdentity_28:output:0"/device:CPU:0*
T0*
_output_shapes
: v
Read_15/DisableCopyOnReadDisableCopyOnRead!read_15_disablecopyonread_count_1"/device:CPU:0*
_output_shapes
 
Read_15/ReadVariableOpReadVariableOp!read_15_disablecopyonread_count_1^Read_15/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0g
Identity_30IdentityRead_15/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: ]
Identity_31IdentityIdentity_30:output:0"/device:CPU:0*
T0*
_output_shapes
: t
Read_16/DisableCopyOnReadDisableCopyOnReadread_16_disablecopyonread_total"/device:CPU:0*
_output_shapes
 
Read_16/ReadVariableOpReadVariableOpread_16_disablecopyonread_total^Read_16/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0g
Identity_32IdentityRead_16/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: ]
Identity_33IdentityIdentity_32:output:0"/device:CPU:0*
T0*
_output_shapes
: t
Read_17/DisableCopyOnReadDisableCopyOnReadread_17_disablecopyonread_count"/device:CPU:0*
_output_shapes
 
Read_17/ReadVariableOpReadVariableOpread_17_disablecopyonread_count^Read_17/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0g
Identity_34IdentityRead_17/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: ]
Identity_35IdentityIdentity_34:output:0"/device:CPU:0*
T0*
_output_shapes
: Н
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*ц
valueмBйB6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB0optimizer/_iterations/.ATTRIBUTES/VARIABLE_VALUEB3optimizer/_learning_rate/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/1/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/2/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/3/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/4/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/5/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/6/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/7/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/8/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*9
value0B.B B B B B B B B B B B B B B B B B B B ё
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0Identity_1:output:0Identity_3:output:0Identity_5:output:0Identity_7:output:0Identity_9:output:0Identity_11:output:0Identity_13:output:0Identity_15:output:0Identity_17:output:0Identity_19:output:0Identity_21:output:0Identity_23:output:0Identity_25:output:0Identity_27:output:0Identity_29:output:0Identity_31:output:0Identity_33:output:0Identity_35:output:0savev2_const"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *!
dtypes
2	
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:Г
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 i
Identity_36Identityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: U
Identity_37IdentityIdentity_36:output:0^NoOp*
T0*
_output_shapes
: §
NoOpNoOp^MergeV2Checkpoints^Read/DisableCopyOnRead^Read/ReadVariableOp^Read_1/DisableCopyOnRead^Read_1/ReadVariableOp^Read_10/DisableCopyOnRead^Read_10/ReadVariableOp^Read_11/DisableCopyOnRead^Read_11/ReadVariableOp^Read_12/DisableCopyOnRead^Read_12/ReadVariableOp^Read_13/DisableCopyOnRead^Read_13/ReadVariableOp^Read_14/DisableCopyOnRead^Read_14/ReadVariableOp^Read_15/DisableCopyOnRead^Read_15/ReadVariableOp^Read_16/DisableCopyOnRead^Read_16/ReadVariableOp^Read_17/DisableCopyOnRead^Read_17/ReadVariableOp^Read_2/DisableCopyOnRead^Read_2/ReadVariableOp^Read_3/DisableCopyOnRead^Read_3/ReadVariableOp^Read_4/DisableCopyOnRead^Read_4/ReadVariableOp^Read_5/DisableCopyOnRead^Read_5/ReadVariableOp^Read_6/DisableCopyOnRead^Read_6/ReadVariableOp^Read_7/DisableCopyOnRead^Read_7/ReadVariableOp^Read_8/DisableCopyOnRead^Read_8/ReadVariableOp^Read_9/DisableCopyOnRead^Read_9/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "#
identity_37Identity_37:output:0*;
_input_shapes*
(: : : : : : : : : : : : : : : : : : : : 2(
MergeV2CheckpointsMergeV2Checkpoints20
Read/DisableCopyOnReadRead/DisableCopyOnRead2*
Read/ReadVariableOpRead/ReadVariableOp24
Read_1/DisableCopyOnReadRead_1/DisableCopyOnRead2.
Read_1/ReadVariableOpRead_1/ReadVariableOp26
Read_10/DisableCopyOnReadRead_10/DisableCopyOnRead20
Read_10/ReadVariableOpRead_10/ReadVariableOp26
Read_11/DisableCopyOnReadRead_11/DisableCopyOnRead20
Read_11/ReadVariableOpRead_11/ReadVariableOp26
Read_12/DisableCopyOnReadRead_12/DisableCopyOnRead20
Read_12/ReadVariableOpRead_12/ReadVariableOp26
Read_13/DisableCopyOnReadRead_13/DisableCopyOnRead20
Read_13/ReadVariableOpRead_13/ReadVariableOp26
Read_14/DisableCopyOnReadRead_14/DisableCopyOnRead20
Read_14/ReadVariableOpRead_14/ReadVariableOp26
Read_15/DisableCopyOnReadRead_15/DisableCopyOnRead20
Read_15/ReadVariableOpRead_15/ReadVariableOp26
Read_16/DisableCopyOnReadRead_16/DisableCopyOnRead20
Read_16/ReadVariableOpRead_16/ReadVariableOp26
Read_17/DisableCopyOnReadRead_17/DisableCopyOnRead20
Read_17/ReadVariableOpRead_17/ReadVariableOp24
Read_2/DisableCopyOnReadRead_2/DisableCopyOnRead2.
Read_2/ReadVariableOpRead_2/ReadVariableOp24
Read_3/DisableCopyOnReadRead_3/DisableCopyOnRead2.
Read_3/ReadVariableOpRead_3/ReadVariableOp24
Read_4/DisableCopyOnReadRead_4/DisableCopyOnRead2.
Read_4/ReadVariableOpRead_4/ReadVariableOp24
Read_5/DisableCopyOnReadRead_5/DisableCopyOnRead2.
Read_5/ReadVariableOpRead_5/ReadVariableOp24
Read_6/DisableCopyOnReadRead_6/DisableCopyOnRead2.
Read_6/ReadVariableOpRead_6/ReadVariableOp24
Read_7/DisableCopyOnReadRead_7/DisableCopyOnRead2.
Read_7/ReadVariableOpRead_7/ReadVariableOp24
Read_8/DisableCopyOnReadRead_8/DisableCopyOnRead2.
Read_8/ReadVariableOpRead_8/ReadVariableOp24
Read_9/DisableCopyOnReadRead_9/DisableCopyOnRead2.
Read_9/ReadVariableOpRead_9/ReadVariableOp:

_output_shapes
: :C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
О
b
F__inference_flatten_57_layer_call_and_return_conditional_losses_283935

inputs
identityV
ConstConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   \
ReshapeReshapeinputsConst:output:0*
T0*'
_output_shapes
:џџџџџџџџџX
IdentityIdentityReshape:output:0*
T0*'
_output_shapes
:џџџџџџџџџ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:џџџџџџџџџ:S O
+
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
э8
Ќ
D__inference_model_57_layer_call_and_return_conditional_losses_283832
inputs_offsource
inputs_onsourceK
5conv1d_53_conv1d_expanddims_1_readvariableop_resource:*7
)conv1d_53_biasadd_readvariableop_resource:@
.injection_masks_matmul_readvariableop_resource:=
/injection_masks_biasadd_readvariableop_resource:
identityЂ&INJECTION_MASKS/BiasAdd/ReadVariableOpЂ%INJECTION_MASKS/MatMul/ReadVariableOpЂ conv1d_53/BiasAdd/ReadVariableOpЂ,conv1d_53/Conv1D/ExpandDims_1/ReadVariableOpШ
%whiten_passthrough_28/PartitionedCallPartitionedCallinputs_offsource*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *<
config_proto,*

CPU

GPU(2*0J

  zE8 *2
f-R+
)__inference_restored_function_body_283383л
reshape_57/PartitionedCallPartitionedCall.whiten_passthrough_28/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *<
config_proto,*

CPU

GPU(2*0J

  zE8 *2
f-R+
)__inference_restored_function_body_283389a
max_pooling1d_69/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :Г
max_pooling1d_69/ExpandDims
ExpandDims#reshape_57/PartitionedCall:output:0(max_pooling1d_69/ExpandDims/dim:output:0*
T0*0
_output_shapes
:џџџџџџџџџЕ
max_pooling1d_69/MaxPoolMaxPool$max_pooling1d_69/ExpandDims:output:0*/
_output_shapes
:џџџџџџџџџb*
ksize
*
paddingSAME*
strides

max_pooling1d_69/SqueezeSqueeze!max_pooling1d_69/MaxPool:output:0*
T0*+
_output_shapes
:џџџџџџџџџb*
squeeze_dims
a
max_pooling1d_70/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :А
max_pooling1d_70/ExpandDims
ExpandDims!max_pooling1d_69/Squeeze:output:0(max_pooling1d_70/ExpandDims/dim:output:0*
T0*/
_output_shapes
:џџџџџџџџџbЕ
max_pooling1d_70/MaxPoolMaxPool$max_pooling1d_70/ExpandDims:output:0*/
_output_shapes
:џџџџџџџџџ*
ksize
*
paddingSAME*
strides

max_pooling1d_70/SqueezeSqueeze!max_pooling1d_70/MaxPool:output:0*
T0*+
_output_shapes
:џџџџџџџџџ*
squeeze_dims
j
conv1d_53/Conv1D/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
§џџџџџџџџА
conv1d_53/Conv1D/ExpandDims
ExpandDims!max_pooling1d_70/Squeeze:output:0(conv1d_53/Conv1D/ExpandDims/dim:output:0*
T0*/
_output_shapes
:џџџџџџџџџІ
,conv1d_53/Conv1D/ExpandDims_1/ReadVariableOpReadVariableOp5conv1d_53_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:**
dtype0c
!conv1d_53/Conv1D/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : О
conv1d_53/Conv1D/ExpandDims_1
ExpandDims4conv1d_53/Conv1D/ExpandDims_1/ReadVariableOp:value:0*conv1d_53/Conv1D/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:*Ъ
conv1d_53/Conv1DConv2D$conv1d_53/Conv1D/ExpandDims:output:0&conv1d_53/Conv1D/ExpandDims_1:output:0*
T0*/
_output_shapes
:џџџџџџџџџ*
paddingSAME*
strides
)
conv1d_53/Conv1D/SqueezeSqueezeconv1d_53/Conv1D:output:0*
T0*+
_output_shapes
:џџџџџџџџџ*
squeeze_dims

§џџџџџџџџ
 conv1d_53/BiasAdd/ReadVariableOpReadVariableOp)conv1d_53_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
conv1d_53/BiasAddBiasAdd!conv1d_53/Conv1D/Squeeze:output:0(conv1d_53/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:џџџџџџџџџS
conv1d_53/betaConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
conv1d_53/mulMulconv1d_53/beta:output:0conv1d_53/BiasAdd:output:0*
T0*+
_output_shapes
:џџџџџџџџџe
conv1d_53/SigmoidSigmoidconv1d_53/mul:z:0*
T0*+
_output_shapes
:џџџџџџџџџ
conv1d_53/mul_1Mulconv1d_53/BiasAdd:output:0conv1d_53/Sigmoid:y:0*
T0*+
_output_shapes
:џџџџџџџџџi
conv1d_53/IdentityIdentityconv1d_53/mul_1:z:0*
T0*+
_output_shapes
:џџџџџџџџџь
conv1d_53/IdentityN	IdentityNconv1d_53/mul_1:z:0conv1d_53/BiasAdd:output:0conv1d_53/beta:output:0*
T
2*,
_gradient_op_typeCustomGradient-283810*D
_output_shapes2
0:џџџџџџџџџ:џџџџџџџџџ: a
max_pooling1d_71/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :Ћ
max_pooling1d_71/ExpandDims
ExpandDimsconv1d_53/IdentityN:output:0(max_pooling1d_71/ExpandDims/dim:output:0*
T0*/
_output_shapes
:џџџџџџџџџЕ
max_pooling1d_71/MaxPoolMaxPool$max_pooling1d_71/ExpandDims:output:0*/
_output_shapes
:џџџџџџџџџ*
ksize
*
paddingSAME*
strides

max_pooling1d_71/SqueezeSqueeze!max_pooling1d_71/MaxPool:output:0*
T0*+
_output_shapes
:џџџџџџџџџ*
squeeze_dims
a
flatten_57/ConstConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   
flatten_57/ReshapeReshape!max_pooling1d_71/Squeeze:output:0flatten_57/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ
%INJECTION_MASKS/MatMul/ReadVariableOpReadVariableOp.injection_masks_matmul_readvariableop_resource*
_output_shapes

:*
dtype0
INJECTION_MASKS/MatMulMatMulflatten_57/Reshape:output:0-INJECTION_MASKS/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ
&INJECTION_MASKS/BiasAdd/ReadVariableOpReadVariableOp/injection_masks_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0І
INJECTION_MASKS/BiasAddBiasAdd INJECTION_MASKS/MatMul:product:0.INJECTION_MASKS/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџv
INJECTION_MASKS/SigmoidSigmoid INJECTION_MASKS/BiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџj
IdentityIdentityINJECTION_MASKS/Sigmoid:y:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџщ
NoOpNoOp'^INJECTION_MASKS/BiasAdd/ReadVariableOp&^INJECTION_MASKS/MatMul/ReadVariableOp!^conv1d_53/BiasAdd/ReadVariableOp-^conv1d_53/Conv1D/ExpandDims_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*L
_input_shapes;
9:џџџџџџџџџ:џџџџџџџџџ : : : : 2P
&INJECTION_MASKS/BiasAdd/ReadVariableOp&INJECTION_MASKS/BiasAdd/ReadVariableOp2N
%INJECTION_MASKS/MatMul/ReadVariableOp%INJECTION_MASKS/MatMul/ReadVariableOp2D
 conv1d_53/BiasAdd/ReadVariableOp conv1d_53/BiasAdd/ReadVariableOp2\
,conv1d_53/Conv1D/ExpandDims_1/ReadVariableOp,conv1d_53/Conv1D/ExpandDims_1/ReadVariableOp:]Y
,
_output_shapes
:џџџџџџџџџ 
)
_user_specified_nameinputs_onsource:_ [
-
_output_shapes
:џџџџџџџџџ
*
_user_specified_nameinputs_offsource
Ъ 
з
D__inference_model_57_layer_call_and_return_conditional_losses_283590
inputs_1

inputs&
conv1d_53_283577:*
conv1d_53_283579:(
injection_masks_283584:$
injection_masks_283586:
identityЂ'INJECTION_MASKS/StatefulPartitionedCallЂ!conv1d_53/StatefulPartitionedCallР
%whiten_passthrough_28/PartitionedCallPartitionedCallinputs_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *<
config_proto,*

CPU

GPU(2*0J

  zE8 *2
f-R+
)__inference_restored_function_body_283383л
reshape_57/PartitionedCallPartitionedCall.whiten_passthrough_28/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *<
config_proto,*

CPU

GPU(2*0J

  zE8 *2
f-R+
)__inference_restored_function_body_283389ј
 max_pooling1d_69/PartitionedCallPartitionedCall#reshape_57/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:џџџџџџџџџb* 
_read_only_resource_inputs
 *<
config_proto,*

CPU

GPU(2*0J

  zE8 *U
fPRN
L__inference_max_pooling1d_69_layer_call_and_return_conditional_losses_283442ў
 max_pooling1d_70/PartitionedCallPartitionedCall)max_pooling1d_69/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *<
config_proto,*

CPU

GPU(2*0J

  zE8 *U
fPRN
L__inference_max_pooling1d_70_layer_call_and_return_conditional_losses_283457Њ
!conv1d_53/StatefulPartitionedCallStatefulPartitionedCall)max_pooling1d_70/PartitionedCall:output:0conv1d_53_283577conv1d_53_283579*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*<
config_proto,*

CPU

GPU(2*0J

  zE8 *N
fIRG
E__inference_conv1d_53_layer_call_and_return_conditional_losses_283511џ
 max_pooling1d_71/PartitionedCallPartitionedCall*conv1d_53/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *<
config_proto,*

CPU

GPU(2*0J

  zE8 *U
fPRN
L__inference_max_pooling1d_71_layer_call_and_return_conditional_losses_283472ю
flatten_57/PartitionedCallPartitionedCall)max_pooling1d_71/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *<
config_proto,*

CPU

GPU(2*0J

  zE8 *O
fJRH
F__inference_flatten_57_layer_call_and_return_conditional_losses_283524И
'INJECTION_MASKS/StatefulPartitionedCallStatefulPartitionedCall#flatten_57/PartitionedCall:output:0injection_masks_283584injection_masks_283586*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*<
config_proto,*

CPU

GPU(2*0J

  zE8 *T
fORM
K__inference_INJECTION_MASKS_layer_call_and_return_conditional_losses_283537
IdentityIdentity0INJECTION_MASKS/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ
NoOpNoOp(^INJECTION_MASKS/StatefulPartitionedCall"^conv1d_53/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*L
_input_shapes;
9:џџџџџџџџџ:џџџџџџџџџ : : : : 2R
'INJECTION_MASKS/StatefulPartitionedCall'INJECTION_MASKS/StatefulPartitionedCall2F
!conv1d_53/StatefulPartitionedCall!conv1d_53/StatefulPartitionedCall:TP
,
_output_shapes
:џџџџџџџџџ 
 
_user_specified_nameinputs:U Q
-
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
о
_
C__inference_reshape_57_layer_call_and_return_conditional_losses_312

inputs
identityc
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          n
	transpose	Transposeinputstranspose/perm:output:0*
T0*,
_output_shapes
:џџџџџџџџџZ
IdentityIdentitytranspose:y:0*
T0*,
_output_shapes
:џџџџџџџџџ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:џџџџџџџџџ:T P
,
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs

M
1__inference_max_pooling1d_70_layer_call_fn_283870

inputs
identityм
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *=
_output_shapes+
):'џџџџџџџџџџџџџџџџџџџџџџџџџџџ* 
_read_only_resource_inputs
 *<
config_proto,*

CPU

GPU(2*0J

  zE8 *U
fPRN
L__inference_max_pooling1d_70_layer_call_and_return_conditional_losses_283457v
IdentityIdentityPartitionedCall:output:0*
T0*=
_output_shapes+
):'џџџџџџџџџџџџџџџџџџџџџџџџџџџ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):'џџџџџџџџџџџџџџџџџџџџџџџџџџџ:e a
=
_output_shapes+
):'џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
 
E
)__inference_restored_function_body_283389

inputs
identityЃ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*,
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *<
config_proto,*

CPU

GPU(2*0J

  zE8 *L
fGRE
C__inference_reshape_57_layer_call_and_return_conditional_losses_512e
IdentityIdentityPartitionedCall:output:0*
T0*,
_output_shapes
:џџџџџџџџџ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:џџџџџџџџџ:T P
,
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs


#__inference_internal_grad_fn_284051
result_grads_0
result_grads_1
result_grads_2
mul_beta
mul_biasadd
identity

identity_1h
mulMulmul_betamul_biasadd^result_grads_0*
T0*+
_output_shapes
:џџџџџџџџџQ
SigmoidSigmoidmul:z:0*
T0*+
_output_shapes
:џџџџџџџџџY
mul_1Mulmul_betamul_biasadd*
T0*+
_output_shapes
:џџџџџџџџџJ
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?]
subSubsub/x:output:0Sigmoid:y:0*
T0*+
_output_shapes
:џџџџџџџџџV
mul_2Mul	mul_1:z:0sub:z:0*
T0*+
_output_shapes
:џџџџџџџџџJ
add/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?]
addAddV2add/x:output:0	mul_2:z:0*
T0*+
_output_shapes
:џџџџџџџџџX
mul_3MulSigmoid:y:0add:z:0*
T0*+
_output_shapes
:џџџџџџџџџS
SquareSquaremul_biasadd*
T0*+
_output_shapes
:џџџџџџџџџ^
mul_4Mulresult_grads_0
Square:y:0*
T0*+
_output_shapes
:џџџџџџџџџZ
mul_5Mul	mul_4:z:0Sigmoid:y:0*
T0*+
_output_shapes
:џџџџџџџџџL
sub_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?a
sub_1Subsub_1/x:output:0Sigmoid:y:0*
T0*+
_output_shapes
:џџџџџџџџџX
mul_6Mul	mul_5:z:0	sub_1:z:0*
T0*+
_output_shapes
:џџџџџџџџџZ
ConstConst*
_output_shapes
:*
dtype0*!
valueB"          F
SumSum	mul_6:z:0Const:output:0*
T0*
_output_shapes
: ]
mul_7Mulresult_grads_0	mul_3:z:0*
T0*+
_output_shapes
:џџџџџџџџџU
IdentityIdentity	mul_7:z:0*
T0*+
_output_shapes
:џџџџџџџџџE

Identity_1IdentitySum:output:0*
T0*
_output_shapes
: "!

identity_1Identity_1:output:0"
identityIdentity:output:0*\
_input_shapesK
I:џџџџџџџџџ:џџџџџџџџџ: : :џџџџџџџџџ:1-
+
_output_shapes
:џџџџџџџџџ:

_output_shapes
: :FB

_output_shapes
: 
(
_user_specified_nameresult_grads_2:[W
+
_output_shapes
:џџџџџџџџџ
(
_user_specified_nameresult_grads_1: 
&
 _has_manual_control_dependencies(
+
_output_shapes
:џџџџџџџџџ
(
_user_specified_nameresult_grads_0

M
1__inference_max_pooling1d_71_layer_call_fn_283916

inputs
identityм
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *=
_output_shapes+
):'џџџџџџџџџџџџџџџџџџџџџџџџџџџ* 
_read_only_resource_inputs
 *<
config_proto,*

CPU

GPU(2*0J

  zE8 *U
fPRN
L__inference_max_pooling1d_71_layer_call_and_return_conditional_losses_283472v
IdentityIdentityPartitionedCall:output:0*
T0*=
_output_shapes+
):'џџџџџџџџџџџџџџџџџџџџџџџџџџџ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):'џџџџџџџџџџџџџџџџџџџџџџџџџџџ:e a
=
_output_shapes+
):'џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
а
h
L__inference_max_pooling1d_70_layer_call_and_return_conditional_losses_283878

inputs
identityP
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :

ExpandDims
ExpandDimsinputsExpandDims/dim:output:0*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџЅ
MaxPoolMaxPoolExpandDims:output:0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ*
ksize
*
paddingSAME*
strides

SqueezeSqueezeMaxPool:output:0*
T0*=
_output_shapes+
):'џџџџџџџџџџџџџџџџџџџџџџџџџџџ*
squeeze_dims
n
IdentityIdentitySqueeze:output:0*
T0*=
_output_shapes+
):'џџџџџџџџџџџџџџџџџџџџџџџџџџџ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):'џџџџџџџџџџџџџџџџџџџџџџџџџџџ:e a
=
_output_shapes+
):'џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
О
D
(__inference_reshape_57_layer_call_fn_317

inputs
identityТ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *<
config_proto,*

CPU

GPU(2*0J

 @E8 *L
fGRE
C__inference_reshape_57_layer_call_and_return_conditional_losses_312e
IdentityIdentityPartitionedCall:output:0*
T0*,
_output_shapes
:џџџџџџџџџ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:џџџџџџџџџ:T P
,
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
ч

*__inference_conv1d_53_layer_call_fn_283887

inputs
unknown:*
	unknown_0:
identityЂStatefulPartitionedCallэ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*<
config_proto,*

CPU

GPU(2*0J

  zE8 *N
fIRG
E__inference_conv1d_53_layer_call_and_return_conditional_losses_283511s
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:џџџџџџџџџ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:џџџџџџџџџ: : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
Ф
S
#__inference__update_step_xla_283837
gradient
variable:**
_XlaMustCompile(*(
_construction_contextkEagerRuntime*#
_input_shapes
:*: *
	_noinline(:($
"
_user_specified_name
variable:L H
"
_output_shapes
:*
"
_user_specified_name
gradient
	
м
$__inference_signature_wrapper_283708
	offsource
onsource
unknown:*
	unknown_0:
	unknown_1:
	unknown_2:
identityЂStatefulPartitionedCallэ
StatefulPartitionedCallStatefulPartitionedCall	offsourceonsourceunknown	unknown_0	unknown_1	unknown_2*
Tin

2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*&
_read_only_resource_inputs
*<
config_proto,*

CPU

GPU(2*0J

  zE8 **
f%R#
!__inference__wrapped_model_283433o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*L
_input_shapes;
9:џџџџџџџџџ:џџџџџџџџџ : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:VR
,
_output_shapes
:џџџџџџџџџ 
"
_user_specified_name
ONSOURCE:X T
-
_output_shapes
:џџџџџџџџџ
#
_user_specified_name	OFFSOURCE
а
h
L__inference_max_pooling1d_71_layer_call_and_return_conditional_losses_283924

inputs
identityP
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :

ExpandDims
ExpandDimsinputsExpandDims/dim:output:0*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџЅ
MaxPoolMaxPoolExpandDims:output:0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ*
ksize
*
paddingSAME*
strides

SqueezeSqueezeMaxPool:output:0*
T0*=
_output_shapes+
):'џџџџџџџџџџџџџџџџџџџџџџџџџџџ*
squeeze_dims
n
IdentityIdentitySqueeze:output:0*
T0*=
_output_shapes+
):'џџџџџџџџџџџџџџџџџџџџџџџџџџџ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):'џџџџџџџџџџџџџџџџџџџџџџџџџџџ:e a
=
_output_shapes+
):'џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
є
j
N__inference_whiten_passthrough_28_layer_call_and_return_conditional_losses_745

inputs
identityВ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_XlaMustCompile(*
_collective_manager_ids
 *,
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *<
config_proto,*

CPU

GPU(2*0J

 @E8 *%
f R
__inference_crop_samples_663I
ShapeShapeinputs*
T0*
_output_shapes
::эЯ]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:б
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:й
strided_slice_1StridedSliceShape:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskR
Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value
B :
Reshape/shapePackstrided_slice:output:0strided_slice_1:output:0Reshape/shape/2:output:0*
N*
T0*
_output_shapes
:{
ReshapeReshapePartitionedCall:output:0Reshape/shape:output:0*
T0*,
_output_shapes
:џџџџџџџџџ]
IdentityIdentityReshape:output:0*
T0*,
_output_shapes
:џџџџџџџџџ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*,
_input_shapes
:џџџџџџџџџ:U Q
-
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
Ъ 
з
D__inference_model_57_layer_call_and_return_conditional_losses_283625
inputs_1

inputs&
conv1d_53_283612:*
conv1d_53_283614:(
injection_masks_283619:$
injection_masks_283621:
identityЂ'INJECTION_MASKS/StatefulPartitionedCallЂ!conv1d_53/StatefulPartitionedCallР
%whiten_passthrough_28/PartitionedCallPartitionedCallinputs_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *<
config_proto,*

CPU

GPU(2*0J

  zE8 *2
f-R+
)__inference_restored_function_body_283383л
reshape_57/PartitionedCallPartitionedCall.whiten_passthrough_28/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *<
config_proto,*

CPU

GPU(2*0J

  zE8 *2
f-R+
)__inference_restored_function_body_283389ј
 max_pooling1d_69/PartitionedCallPartitionedCall#reshape_57/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:џџџџџџџџџb* 
_read_only_resource_inputs
 *<
config_proto,*

CPU

GPU(2*0J

  zE8 *U
fPRN
L__inference_max_pooling1d_69_layer_call_and_return_conditional_losses_283442ў
 max_pooling1d_70/PartitionedCallPartitionedCall)max_pooling1d_69/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *<
config_proto,*

CPU

GPU(2*0J

  zE8 *U
fPRN
L__inference_max_pooling1d_70_layer_call_and_return_conditional_losses_283457Њ
!conv1d_53/StatefulPartitionedCallStatefulPartitionedCall)max_pooling1d_70/PartitionedCall:output:0conv1d_53_283612conv1d_53_283614*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*<
config_proto,*

CPU

GPU(2*0J

  zE8 *N
fIRG
E__inference_conv1d_53_layer_call_and_return_conditional_losses_283511џ
 max_pooling1d_71/PartitionedCallPartitionedCall*conv1d_53/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *<
config_proto,*

CPU

GPU(2*0J

  zE8 *U
fPRN
L__inference_max_pooling1d_71_layer_call_and_return_conditional_losses_283472ю
flatten_57/PartitionedCallPartitionedCall)max_pooling1d_71/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *<
config_proto,*

CPU

GPU(2*0J

  zE8 *O
fJRH
F__inference_flatten_57_layer_call_and_return_conditional_losses_283524И
'INJECTION_MASKS/StatefulPartitionedCallStatefulPartitionedCall#flatten_57/PartitionedCall:output:0injection_masks_283619injection_masks_283621*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*<
config_proto,*

CPU

GPU(2*0J

  zE8 *T
fORM
K__inference_INJECTION_MASKS_layer_call_and_return_conditional_losses_283537
IdentityIdentity0INJECTION_MASKS/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ
NoOpNoOp(^INJECTION_MASKS/StatefulPartitionedCall"^conv1d_53/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*L
_input_shapes;
9:џџџџџџџџџ:џџџџџџџџџ : : : : 2R
'INJECTION_MASKS/StatefulPartitionedCall'INJECTION_MASKS/StatefulPartitionedCall2F
!conv1d_53/StatefulPartitionedCall!conv1d_53/StatefulPartitionedCall:TP
,
_output_shapes
:џџџџџџџџџ 
 
_user_specified_nameinputs:U Q
-
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
Ч	
с
)__inference_model_57_layer_call_fn_283601
	offsource
onsource
unknown:*
	unknown_0:
	unknown_1:
	unknown_2:
identityЂStatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCall	offsourceonsourceunknown	unknown_0	unknown_1	unknown_2*
Tin

2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*&
_read_only_resource_inputs
*<
config_proto,*

CPU

GPU(2*0J

  zE8 *M
fHRF
D__inference_model_57_layer_call_and_return_conditional_losses_283590o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*L
_input_shapes;
9:џџџџџџџџџ:џџџџџџџџџ : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:VR
,
_output_shapes
:џџџџџџџџџ 
"
_user_specified_name
ONSOURCE:X T
-
_output_shapes
:џџџџџџџџџ
#
_user_specified_name	OFFSOURCE
Ѕ?
У
!__inference__wrapped_model_283433
	offsource
onsourceT
>model_57_conv1d_53_conv1d_expanddims_1_readvariableop_resource:*@
2model_57_conv1d_53_biasadd_readvariableop_resource:I
7model_57_injection_masks_matmul_readvariableop_resource:F
8model_57_injection_masks_biasadd_readvariableop_resource:
identityЂ/model_57/INJECTION_MASKS/BiasAdd/ReadVariableOpЂ.model_57/INJECTION_MASKS/MatMul/ReadVariableOpЂ)model_57/conv1d_53/BiasAdd/ReadVariableOpЂ5model_57/conv1d_53/Conv1D/ExpandDims_1/ReadVariableOpЪ
.model_57/whiten_passthrough_28/PartitionedCallPartitionedCall	offsource*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *<
config_proto,*

CPU

GPU(2*0J

  zE8 *2
f-R+
)__inference_restored_function_body_283383э
#model_57/reshape_57/PartitionedCallPartitionedCall7model_57/whiten_passthrough_28/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *<
config_proto,*

CPU

GPU(2*0J

  zE8 *2
f-R+
)__inference_restored_function_body_283389j
(model_57/max_pooling1d_69/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :Ю
$model_57/max_pooling1d_69/ExpandDims
ExpandDims,model_57/reshape_57/PartitionedCall:output:01model_57/max_pooling1d_69/ExpandDims/dim:output:0*
T0*0
_output_shapes
:џџџџџџџџџЧ
!model_57/max_pooling1d_69/MaxPoolMaxPool-model_57/max_pooling1d_69/ExpandDims:output:0*/
_output_shapes
:џџџџџџџџџb*
ksize
*
paddingSAME*
strides
Ѕ
!model_57/max_pooling1d_69/SqueezeSqueeze*model_57/max_pooling1d_69/MaxPool:output:0*
T0*+
_output_shapes
:џџџџџџџџџb*
squeeze_dims
j
(model_57/max_pooling1d_70/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :Ы
$model_57/max_pooling1d_70/ExpandDims
ExpandDims*model_57/max_pooling1d_69/Squeeze:output:01model_57/max_pooling1d_70/ExpandDims/dim:output:0*
T0*/
_output_shapes
:џџџџџџџџџbЧ
!model_57/max_pooling1d_70/MaxPoolMaxPool-model_57/max_pooling1d_70/ExpandDims:output:0*/
_output_shapes
:џџџџџџџџџ*
ksize
*
paddingSAME*
strides
Ѕ
!model_57/max_pooling1d_70/SqueezeSqueeze*model_57/max_pooling1d_70/MaxPool:output:0*
T0*+
_output_shapes
:џџџџџџџџџ*
squeeze_dims
s
(model_57/conv1d_53/Conv1D/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
§џџџџџџџџЫ
$model_57/conv1d_53/Conv1D/ExpandDims
ExpandDims*model_57/max_pooling1d_70/Squeeze:output:01model_57/conv1d_53/Conv1D/ExpandDims/dim:output:0*
T0*/
_output_shapes
:џџџџџџџџџИ
5model_57/conv1d_53/Conv1D/ExpandDims_1/ReadVariableOpReadVariableOp>model_57_conv1d_53_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:**
dtype0l
*model_57/conv1d_53/Conv1D/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : й
&model_57/conv1d_53/Conv1D/ExpandDims_1
ExpandDims=model_57/conv1d_53/Conv1D/ExpandDims_1/ReadVariableOp:value:03model_57/conv1d_53/Conv1D/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:*х
model_57/conv1d_53/Conv1DConv2D-model_57/conv1d_53/Conv1D/ExpandDims:output:0/model_57/conv1d_53/Conv1D/ExpandDims_1:output:0*
T0*/
_output_shapes
:џџџџџџџџџ*
paddingSAME*
strides
)І
!model_57/conv1d_53/Conv1D/SqueezeSqueeze"model_57/conv1d_53/Conv1D:output:0*
T0*+
_output_shapes
:џџџџџџџџџ*
squeeze_dims

§џџџџџџџџ
)model_57/conv1d_53/BiasAdd/ReadVariableOpReadVariableOp2model_57_conv1d_53_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0К
model_57/conv1d_53/BiasAddBiasAdd*model_57/conv1d_53/Conv1D/Squeeze:output:01model_57/conv1d_53/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:џџџџџџџџџ\
model_57/conv1d_53/betaConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
model_57/conv1d_53/mulMul model_57/conv1d_53/beta:output:0#model_57/conv1d_53/BiasAdd:output:0*
T0*+
_output_shapes
:џџџџџџџџџw
model_57/conv1d_53/SigmoidSigmoidmodel_57/conv1d_53/mul:z:0*
T0*+
_output_shapes
:џџџџџџџџџ
model_57/conv1d_53/mul_1Mul#model_57/conv1d_53/BiasAdd:output:0model_57/conv1d_53/Sigmoid:y:0*
T0*+
_output_shapes
:џџџџџџџџџ{
model_57/conv1d_53/IdentityIdentitymodel_57/conv1d_53/mul_1:z:0*
T0*+
_output_shapes
:џџџџџџџџџ
model_57/conv1d_53/IdentityN	IdentityNmodel_57/conv1d_53/mul_1:z:0#model_57/conv1d_53/BiasAdd:output:0 model_57/conv1d_53/beta:output:0*
T
2*,
_gradient_op_typeCustomGradient-283411*D
_output_shapes2
0:џџџџџџџџџ:џџџџџџџџџ: j
(model_57/max_pooling1d_71/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :Ц
$model_57/max_pooling1d_71/ExpandDims
ExpandDims%model_57/conv1d_53/IdentityN:output:01model_57/max_pooling1d_71/ExpandDims/dim:output:0*
T0*/
_output_shapes
:џџџџџџџџџЧ
!model_57/max_pooling1d_71/MaxPoolMaxPool-model_57/max_pooling1d_71/ExpandDims:output:0*/
_output_shapes
:џџџџџџџџџ*
ksize
*
paddingSAME*
strides
Ѕ
!model_57/max_pooling1d_71/SqueezeSqueeze*model_57/max_pooling1d_71/MaxPool:output:0*
T0*+
_output_shapes
:џџџџџџџџџ*
squeeze_dims
j
model_57/flatten_57/ConstConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   Ј
model_57/flatten_57/ReshapeReshape*model_57/max_pooling1d_71/Squeeze:output:0"model_57/flatten_57/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџІ
.model_57/INJECTION_MASKS/MatMul/ReadVariableOpReadVariableOp7model_57_injection_masks_matmul_readvariableop_resource*
_output_shapes

:*
dtype0Й
model_57/INJECTION_MASKS/MatMulMatMul$model_57/flatten_57/Reshape:output:06model_57/INJECTION_MASKS/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџЄ
/model_57/INJECTION_MASKS/BiasAdd/ReadVariableOpReadVariableOp8model_57_injection_masks_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0С
 model_57/INJECTION_MASKS/BiasAddBiasAdd)model_57/INJECTION_MASKS/MatMul:product:07model_57/INJECTION_MASKS/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ
 model_57/INJECTION_MASKS/SigmoidSigmoid)model_57/INJECTION_MASKS/BiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџs
IdentityIdentity$model_57/INJECTION_MASKS/Sigmoid:y:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ
NoOpNoOp0^model_57/INJECTION_MASKS/BiasAdd/ReadVariableOp/^model_57/INJECTION_MASKS/MatMul/ReadVariableOp*^model_57/conv1d_53/BiasAdd/ReadVariableOp6^model_57/conv1d_53/Conv1D/ExpandDims_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*L
_input_shapes;
9:џџџџџџџџџ:џџџџџџџџџ : : : : 2b
/model_57/INJECTION_MASKS/BiasAdd/ReadVariableOp/model_57/INJECTION_MASKS/BiasAdd/ReadVariableOp2`
.model_57/INJECTION_MASKS/MatMul/ReadVariableOp.model_57/INJECTION_MASKS/MatMul/ReadVariableOp2V
)model_57/conv1d_53/BiasAdd/ReadVariableOp)model_57/conv1d_53/BiasAdd/ReadVariableOp2n
5model_57/conv1d_53/Conv1D/ExpandDims_1/ReadVariableOp5model_57/conv1d_53/Conv1D/ExpandDims_1/ReadVariableOp:VR
,
_output_shapes
:џџџџџџџџџ 
"
_user_specified_name
ONSOURCE:X T
-
_output_shapes
:џџџџџџџџџ
#
_user_specified_name	OFFSOURCE
 
Ф
#__inference_internal_grad_fn_284135
result_grads_0
result_grads_1
result_grads_2
mul_model_57_conv1d_53_beta"
mul_model_57_conv1d_53_biasadd
identity

identity_1
mulMulmul_model_57_conv1d_53_betamul_model_57_conv1d_53_biasadd^result_grads_0*
T0*+
_output_shapes
:џџџџџџџџџQ
SigmoidSigmoidmul:z:0*
T0*+
_output_shapes
:џџџџџџџџџ
mul_1Mulmul_model_57_conv1d_53_betamul_model_57_conv1d_53_biasadd*
T0*+
_output_shapes
:џџџџџџџџџJ
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?]
subSubsub/x:output:0Sigmoid:y:0*
T0*+
_output_shapes
:џџџџџџџџџV
mul_2Mul	mul_1:z:0sub:z:0*
T0*+
_output_shapes
:џџџџџџџџџJ
add/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?]
addAddV2add/x:output:0	mul_2:z:0*
T0*+
_output_shapes
:џџџџџџџџџX
mul_3MulSigmoid:y:0add:z:0*
T0*+
_output_shapes
:џџџџџџџџџf
SquareSquaremul_model_57_conv1d_53_biasadd*
T0*+
_output_shapes
:џџџџџџџџџ^
mul_4Mulresult_grads_0
Square:y:0*
T0*+
_output_shapes
:џџџџџџџџџZ
mul_5Mul	mul_4:z:0Sigmoid:y:0*
T0*+
_output_shapes
:џџџџџџџџџL
sub_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?a
sub_1Subsub_1/x:output:0Sigmoid:y:0*
T0*+
_output_shapes
:џџџџџџџџџX
mul_6Mul	mul_5:z:0	sub_1:z:0*
T0*+
_output_shapes
:џџџџџџџџџZ
ConstConst*
_output_shapes
:*
dtype0*!
valueB"          F
SumSum	mul_6:z:0Const:output:0*
T0*
_output_shapes
: ]
mul_7Mulresult_grads_0	mul_3:z:0*
T0*+
_output_shapes
:џџџџџџџџџU
IdentityIdentity	mul_7:z:0*
T0*+
_output_shapes
:џџџџџџџџџE

Identity_1IdentitySum:output:0*
T0*
_output_shapes
: "!

identity_1Identity_1:output:0"
identityIdentity:output:0*\
_input_shapesK
I:џџџџџџџџџ:џџџџџџџџџ: : :џџџџџџџџџ:1-
+
_output_shapes
:џџџџџџџџџ:

_output_shapes
: :FB

_output_shapes
: 
(
_user_specified_nameresult_grads_2:[W
+
_output_shapes
:џџџџџџџџџ
(
_user_specified_nameresult_grads_1: 
&
 _has_manual_control_dependencies(
+
_output_shapes
:џџџџџџџџџ
(
_user_specified_nameresult_grads_0
Ќ
K
#__inference__update_step_xla_283842
gradient
variable:*
_XlaMustCompile(*(
_construction_contextkEagerRuntime*
_input_shapes

:: *
	_noinline(:($
"
_user_specified_name
variable:D @

_output_shapes
:
"
_user_specified_name
gradient
э8
Ќ
D__inference_model_57_layer_call_and_return_conditional_losses_283784
inputs_offsource
inputs_onsourceK
5conv1d_53_conv1d_expanddims_1_readvariableop_resource:*7
)conv1d_53_biasadd_readvariableop_resource:@
.injection_masks_matmul_readvariableop_resource:=
/injection_masks_biasadd_readvariableop_resource:
identityЂ&INJECTION_MASKS/BiasAdd/ReadVariableOpЂ%INJECTION_MASKS/MatMul/ReadVariableOpЂ conv1d_53/BiasAdd/ReadVariableOpЂ,conv1d_53/Conv1D/ExpandDims_1/ReadVariableOpШ
%whiten_passthrough_28/PartitionedCallPartitionedCallinputs_offsource*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *<
config_proto,*

CPU

GPU(2*0J

  zE8 *2
f-R+
)__inference_restored_function_body_283383л
reshape_57/PartitionedCallPartitionedCall.whiten_passthrough_28/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *<
config_proto,*

CPU

GPU(2*0J

  zE8 *2
f-R+
)__inference_restored_function_body_283389a
max_pooling1d_69/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :Г
max_pooling1d_69/ExpandDims
ExpandDims#reshape_57/PartitionedCall:output:0(max_pooling1d_69/ExpandDims/dim:output:0*
T0*0
_output_shapes
:џџџџџџџџџЕ
max_pooling1d_69/MaxPoolMaxPool$max_pooling1d_69/ExpandDims:output:0*/
_output_shapes
:џџџџџџџџџb*
ksize
*
paddingSAME*
strides

max_pooling1d_69/SqueezeSqueeze!max_pooling1d_69/MaxPool:output:0*
T0*+
_output_shapes
:џџџџџџџџџb*
squeeze_dims
a
max_pooling1d_70/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :А
max_pooling1d_70/ExpandDims
ExpandDims!max_pooling1d_69/Squeeze:output:0(max_pooling1d_70/ExpandDims/dim:output:0*
T0*/
_output_shapes
:џџџџџџџџџbЕ
max_pooling1d_70/MaxPoolMaxPool$max_pooling1d_70/ExpandDims:output:0*/
_output_shapes
:џџџџџџџџџ*
ksize
*
paddingSAME*
strides

max_pooling1d_70/SqueezeSqueeze!max_pooling1d_70/MaxPool:output:0*
T0*+
_output_shapes
:џџџџџџџџџ*
squeeze_dims
j
conv1d_53/Conv1D/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
§џџџџџџџџА
conv1d_53/Conv1D/ExpandDims
ExpandDims!max_pooling1d_70/Squeeze:output:0(conv1d_53/Conv1D/ExpandDims/dim:output:0*
T0*/
_output_shapes
:џџџџџџџџџІ
,conv1d_53/Conv1D/ExpandDims_1/ReadVariableOpReadVariableOp5conv1d_53_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:**
dtype0c
!conv1d_53/Conv1D/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : О
conv1d_53/Conv1D/ExpandDims_1
ExpandDims4conv1d_53/Conv1D/ExpandDims_1/ReadVariableOp:value:0*conv1d_53/Conv1D/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:*Ъ
conv1d_53/Conv1DConv2D$conv1d_53/Conv1D/ExpandDims:output:0&conv1d_53/Conv1D/ExpandDims_1:output:0*
T0*/
_output_shapes
:џџџџџџџџџ*
paddingSAME*
strides
)
conv1d_53/Conv1D/SqueezeSqueezeconv1d_53/Conv1D:output:0*
T0*+
_output_shapes
:џџџџџџџџџ*
squeeze_dims

§џџџџџџџџ
 conv1d_53/BiasAdd/ReadVariableOpReadVariableOp)conv1d_53_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
conv1d_53/BiasAddBiasAdd!conv1d_53/Conv1D/Squeeze:output:0(conv1d_53/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:џџџџџџџџџS
conv1d_53/betaConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
conv1d_53/mulMulconv1d_53/beta:output:0conv1d_53/BiasAdd:output:0*
T0*+
_output_shapes
:џџџџџџџџџe
conv1d_53/SigmoidSigmoidconv1d_53/mul:z:0*
T0*+
_output_shapes
:џџџџџџџџџ
conv1d_53/mul_1Mulconv1d_53/BiasAdd:output:0conv1d_53/Sigmoid:y:0*
T0*+
_output_shapes
:џџџџџџџџџi
conv1d_53/IdentityIdentityconv1d_53/mul_1:z:0*
T0*+
_output_shapes
:џџџџџџџџџь
conv1d_53/IdentityN	IdentityNconv1d_53/mul_1:z:0conv1d_53/BiasAdd:output:0conv1d_53/beta:output:0*
T
2*,
_gradient_op_typeCustomGradient-283762*D
_output_shapes2
0:џџџџџџџџџ:џџџџџџџџџ: a
max_pooling1d_71/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :Ћ
max_pooling1d_71/ExpandDims
ExpandDimsconv1d_53/IdentityN:output:0(max_pooling1d_71/ExpandDims/dim:output:0*
T0*/
_output_shapes
:џџџџџџџџџЕ
max_pooling1d_71/MaxPoolMaxPool$max_pooling1d_71/ExpandDims:output:0*/
_output_shapes
:џџџџџџџџџ*
ksize
*
paddingSAME*
strides

max_pooling1d_71/SqueezeSqueeze!max_pooling1d_71/MaxPool:output:0*
T0*+
_output_shapes
:џџџџџџџџџ*
squeeze_dims
a
flatten_57/ConstConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   
flatten_57/ReshapeReshape!max_pooling1d_71/Squeeze:output:0flatten_57/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ
%INJECTION_MASKS/MatMul/ReadVariableOpReadVariableOp.injection_masks_matmul_readvariableop_resource*
_output_shapes

:*
dtype0
INJECTION_MASKS/MatMulMatMulflatten_57/Reshape:output:0-INJECTION_MASKS/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ
&INJECTION_MASKS/BiasAdd/ReadVariableOpReadVariableOp/injection_masks_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0І
INJECTION_MASKS/BiasAddBiasAdd INJECTION_MASKS/MatMul:product:0.INJECTION_MASKS/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџv
INJECTION_MASKS/SigmoidSigmoid INJECTION_MASKS/BiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџj
IdentityIdentityINJECTION_MASKS/Sigmoid:y:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџщ
NoOpNoOp'^INJECTION_MASKS/BiasAdd/ReadVariableOp&^INJECTION_MASKS/MatMul/ReadVariableOp!^conv1d_53/BiasAdd/ReadVariableOp-^conv1d_53/Conv1D/ExpandDims_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*L
_input_shapes;
9:џџџџџџџџџ:џџџџџџџџџ : : : : 2P
&INJECTION_MASKS/BiasAdd/ReadVariableOp&INJECTION_MASKS/BiasAdd/ReadVariableOp2N
%INJECTION_MASKS/MatMul/ReadVariableOp%INJECTION_MASKS/MatMul/ReadVariableOp2D
 conv1d_53/BiasAdd/ReadVariableOp conv1d_53/BiasAdd/ReadVariableOp2\
,conv1d_53/Conv1D/ExpandDims_1/ReadVariableOp,conv1d_53/Conv1D/ExpandDims_1/ReadVariableOp:]Y
,
_output_shapes
:џџџџџџџџџ 
)
_user_specified_nameinputs_onsource:_ [
-
_output_shapes
:џџџџџџџџџ
*
_user_specified_nameinputs_offsource
Ё

ќ
K__inference_INJECTION_MASKS_layer_call_and_return_conditional_losses_283955

inputs0
matmul_readvariableop_resource:-
biasadd_readvariableop_resource:
identityЂBiasAdd/ReadVariableOpЂMatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџV
SigmoidSigmoidBiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџZ
IdentityIdentitySigmoid:y:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџw
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:џџџџџџџџџ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs<
#__inference_internal_grad_fn_284023CustomGradient-283902<
#__inference_internal_grad_fn_284051CustomGradient-283502<
#__inference_internal_grad_fn_284079CustomGradient-283762<
#__inference_internal_grad_fn_284107CustomGradient-283810<
#__inference_internal_grad_fn_284135CustomGradient-283411"ѓ
L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*
serving_defaultь
E
	OFFSOURCE8
serving_default_OFFSOURCE:0џџџџџџџџџ
B
ONSOURCE6
serving_default_ONSOURCE:0џџџџџџџџџ C
INJECTION_MASKS0
StatefulPartitionedCall:0џџџџџџџџџtensorflow/serving/predict:јю
О
layer-0
layer-1
layer-2
layer-3
layer-4
layer_with_weights-0
layer-5
layer-6
layer-7
	layer-8

layer_with_weights-1

layer-9
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
_default_save_signature
	optimizer

signatures
#_self_saveable_object_factories"
_tf_keras_network
D
#_self_saveable_object_factories"
_tf_keras_input_layer
Ъ
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
#_self_saveable_object_factories"
_tf_keras_layer
Ъ
	variables
trainable_variables
regularization_losses
 	keras_api
!__call__
*"&call_and_return_all_conditional_losses
##_self_saveable_object_factories"
_tf_keras_layer
Ъ
$	variables
%trainable_variables
&regularization_losses
'	keras_api
(__call__
*)&call_and_return_all_conditional_losses
#*_self_saveable_object_factories"
_tf_keras_layer
Ъ
+	variables
,trainable_variables
-regularization_losses
.	keras_api
/__call__
*0&call_and_return_all_conditional_losses
#1_self_saveable_object_factories"
_tf_keras_layer

2	variables
3trainable_variables
4regularization_losses
5	keras_api
6__call__
*7&call_and_return_all_conditional_losses

8kernel
9bias
#:_self_saveable_object_factories
 ;_jit_compiled_convolution_op"
_tf_keras_layer
Ъ
<	variables
=trainable_variables
>regularization_losses
?	keras_api
@__call__
*A&call_and_return_all_conditional_losses
#B_self_saveable_object_factories"
_tf_keras_layer
Ъ
C	variables
Dtrainable_variables
Eregularization_losses
F	keras_api
G__call__
*H&call_and_return_all_conditional_losses
#I_self_saveable_object_factories"
_tf_keras_layer
D
#J_self_saveable_object_factories"
_tf_keras_input_layer
р
K	variables
Ltrainable_variables
Mregularization_losses
N	keras_api
O__call__
*P&call_and_return_all_conditional_losses

Qkernel
Rbias
#S_self_saveable_object_factories"
_tf_keras_layer
<
80
91
Q2
R3"
trackable_list_wrapper
<
80
91
Q2
R3"
trackable_list_wrapper
 "
trackable_list_wrapper
Ъ
Tnon_trainable_variables

Ulayers
Vmetrics
Wlayer_regularization_losses
Xlayer_metrics
	variables
trainable_variables
regularization_losses
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
Я
Ytrace_0
Ztrace_1
[trace_2
\trace_32ф
)__inference_model_57_layer_call_fn_283601
)__inference_model_57_layer_call_fn_283636
)__inference_model_57_layer_call_fn_283722
)__inference_model_57_layer_call_fn_283736Е
ЎВЊ
FullArgSpec)
args!
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЂ
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 zYtrace_0zZtrace_1z[trace_2z\trace_3
Л
]trace_0
^trace_1
_trace_2
`trace_32а
D__inference_model_57_layer_call_and_return_conditional_losses_283544
D__inference_model_57_layer_call_and_return_conditional_losses_283565
D__inference_model_57_layer_call_and_return_conditional_losses_283784
D__inference_model_57_layer_call_and_return_conditional_losses_283832Е
ЎВЊ
FullArgSpec)
args!
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЂ
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 z]trace_0z^trace_1z_trace_2z`trace_3
иBе
!__inference__wrapped_model_283433	OFFSOURCEONSOURCE"
В
FullArgSpec
args 
varargsjargs
varkwjkwargs
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 

a
_variables
b_iterations
c_learning_rate
d_index_dict
e
_momentums
f_velocities
g_update_step_xla"
experimentalOptimizer
,
hserving_default"
signature_map
 "
trackable_dict_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
­
inon_trainable_variables

jlayers
kmetrics
llayer_regularization_losses
mlayer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
э
ntrace_02а
3__inference_whiten_passthrough_28_layer_call_fn_750
В
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 zntrace_0

otrace_02ы
N__inference_whiten_passthrough_28_layer_call_and_return_conditional_losses_767
В
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 zotrace_0
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
­
pnon_trainable_variables

qlayers
rmetrics
slayer_regularization_losses
tlayer_metrics
	variables
trainable_variables
regularization_losses
!__call__
*"&call_and_return_all_conditional_losses
&""call_and_return_conditional_losses"
_generic_user_object
т
utrace_02Х
(__inference_reshape_57_layer_call_fn_317
В
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 zutrace_0
§
vtrace_02р
C__inference_reshape_57_layer_call_and_return_conditional_losses_512
В
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 zvtrace_0
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
­
wnon_trainable_variables

xlayers
ymetrics
zlayer_regularization_losses
{layer_metrics
$	variables
%trainable_variables
&regularization_losses
(__call__
*)&call_and_return_all_conditional_losses
&)"call_and_return_conditional_losses"
_generic_user_object
ы
|trace_02Ю
1__inference_max_pooling1d_69_layer_call_fn_283857
В
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 z|trace_0

}trace_02щ
L__inference_max_pooling1d_69_layer_call_and_return_conditional_losses_283865
В
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 z}trace_0
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
А
~non_trainable_variables

layers
metrics
 layer_regularization_losses
layer_metrics
+	variables
,trainable_variables
-regularization_losses
/__call__
*0&call_and_return_all_conditional_losses
&0"call_and_return_conditional_losses"
_generic_user_object
э
trace_02Ю
1__inference_max_pooling1d_70_layer_call_fn_283870
В
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 ztrace_0

trace_02щ
L__inference_max_pooling1d_70_layer_call_and_return_conditional_losses_283878
В
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 ztrace_0
 "
trackable_dict_wrapper
.
80
91"
trackable_list_wrapper
.
80
91"
trackable_list_wrapper
 "
trackable_list_wrapper
В
non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
2	variables
3trainable_variables
4regularization_losses
6__call__
*7&call_and_return_all_conditional_losses
&7"call_and_return_conditional_losses"
_generic_user_object
ц
trace_02Ч
*__inference_conv1d_53_layer_call_fn_283887
В
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 ztrace_0

trace_02т
E__inference_conv1d_53_layer_call_and_return_conditional_losses_283911
В
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 ztrace_0
:* 2kernel
: 2bias
 "
trackable_dict_wrapper
Њ2ЇЄ
В
FullArgSpec
args
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
В
non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
<	variables
=trainable_variables
>regularization_losses
@__call__
*A&call_and_return_all_conditional_losses
&A"call_and_return_conditional_losses"
_generic_user_object
э
trace_02Ю
1__inference_max_pooling1d_71_layer_call_fn_283916
В
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 ztrace_0

trace_02щ
L__inference_max_pooling1d_71_layer_call_and_return_conditional_losses_283924
В
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 ztrace_0
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
В
non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
C	variables
Dtrainable_variables
Eregularization_losses
G__call__
*H&call_and_return_all_conditional_losses
&H"call_and_return_conditional_losses"
_generic_user_object
ч
trace_02Ш
+__inference_flatten_57_layer_call_fn_283929
В
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 ztrace_0

trace_02у
F__inference_flatten_57_layer_call_and_return_conditional_losses_283935
В
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 ztrace_0
 "
trackable_dict_wrapper
 "
trackable_dict_wrapper
.
Q0
R1"
trackable_list_wrapper
.
Q0
R1"
trackable_list_wrapper
 "
trackable_list_wrapper
В
non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
K	variables
Ltrainable_variables
Mregularization_losses
O__call__
*P&call_and_return_all_conditional_losses
&P"call_and_return_conditional_losses"
_generic_user_object
ь
trace_02Э
0__inference_INJECTION_MASKS_layer_call_fn_283944
В
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 ztrace_0

 trace_02ш
K__inference_INJECTION_MASKS_layer_call_and_return_conditional_losses_283955
В
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 z trace_0
: 2kernel
: 2bias
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
f
0
1
2
3
4
5
6
7
	8

9"
trackable_list_wrapper
0
Ё0
Ђ1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
§Bњ
)__inference_model_57_layer_call_fn_283601	OFFSOURCEONSOURCE"Е
ЎВЊ
FullArgSpec)
args!
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЂ
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
§Bњ
)__inference_model_57_layer_call_fn_283636	OFFSOURCEONSOURCE"Е
ЎВЊ
FullArgSpec)
args!
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЂ
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
B
)__inference_model_57_layer_call_fn_283722inputs_offsourceinputs_onsource"Е
ЎВЊ
FullArgSpec)
args!
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЂ
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
B
)__inference_model_57_layer_call_fn_283736inputs_offsourceinputs_onsource"Е
ЎВЊ
FullArgSpec)
args!
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЂ
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
B
D__inference_model_57_layer_call_and_return_conditional_losses_283544	OFFSOURCEONSOURCE"Е
ЎВЊ
FullArgSpec)
args!
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЂ
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
B
D__inference_model_57_layer_call_and_return_conditional_losses_283565	OFFSOURCEONSOURCE"Е
ЎВЊ
FullArgSpec)
args!
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЂ
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
ІBЃ
D__inference_model_57_layer_call_and_return_conditional_losses_283784inputs_offsourceinputs_onsource"Е
ЎВЊ
FullArgSpec)
args!
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЂ
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
ІBЃ
D__inference_model_57_layer_call_and_return_conditional_losses_283832inputs_offsourceinputs_onsource"Е
ЎВЊ
FullArgSpec)
args!
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЂ
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
g
b0
Ѓ1
Є2
Ѕ3
І4
Ї5
Ј6
Љ7
Њ8"
trackable_list_wrapper
:	  2	iteration
:  2learning_rate
 "
trackable_dict_wrapper
@
Ѓ0
Ѕ1
Ї2
Љ3"
trackable_list_wrapper
@
Є0
І1
Ј2
Њ3"
trackable_list_wrapper
Й
Ћtrace_0
Ќtrace_1
­trace_2
Ўtrace_32Ц
#__inference__update_step_xla_283837
#__inference__update_step_xla_283842
#__inference__update_step_xla_283847
#__inference__update_step_xla_283852Џ
ІВЂ
FullArgSpec*
args"

jgradient

jvariable
jkey
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 0zЋtrace_0zЌtrace_1z­trace_2zЎtrace_3
еBв
$__inference_signature_wrapper_283708	OFFSOURCEONSOURCE"
В
FullArgSpec
args 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
нBк
3__inference_whiten_passthrough_28_layer_call_fn_750inputs"
В
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
јBѕ
N__inference_whiten_passthrough_28_layer_call_and_return_conditional_losses_767inputs"
В
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
вBЯ
(__inference_reshape_57_layer_call_fn_317inputs"
В
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
эBъ
C__inference_reshape_57_layer_call_and_return_conditional_losses_512inputs"
В
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
лBи
1__inference_max_pooling1d_69_layer_call_fn_283857inputs"
В
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
іBѓ
L__inference_max_pooling1d_69_layer_call_and_return_conditional_losses_283865inputs"
В
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
лBи
1__inference_max_pooling1d_70_layer_call_fn_283870inputs"
В
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
іBѓ
L__inference_max_pooling1d_70_layer_call_and_return_conditional_losses_283878inputs"
В
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
дBб
*__inference_conv1d_53_layer_call_fn_283887inputs"
В
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
яBь
E__inference_conv1d_53_layer_call_and_return_conditional_losses_283911inputs"
В
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
лBи
1__inference_max_pooling1d_71_layer_call_fn_283916inputs"
В
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
іBѓ
L__inference_max_pooling1d_71_layer_call_and_return_conditional_losses_283924inputs"
В
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
еBв
+__inference_flatten_57_layer_call_fn_283929inputs"
В
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
№Bэ
F__inference_flatten_57_layer_call_and_return_conditional_losses_283935inputs"
В
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
кBз
0__inference_INJECTION_MASKS_layer_call_fn_283944inputs"
В
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
ѕBђ
K__inference_INJECTION_MASKS_layer_call_and_return_conditional_losses_283955inputs"
В
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
R
Џ	variables
А	keras_api

Бtotal

Вcount"
_tf_keras_metric
c
Г	variables
Д	keras_api

Еtotal

Жcount
З
_fn_kwargs"
_tf_keras_metric
#:!* 2Adam/m/kernel
#:!* 2Adam/v/kernel
: 2Adam/m/bias
: 2Adam/v/bias
: 2Adam/m/kernel
: 2Adam/v/kernel
: 2Adam/m/bias
: 2Adam/v/bias
юBы
#__inference__update_step_xla_283837gradientvariable"­
ІВЂ
FullArgSpec*
args"

jgradient

jvariable
jkey
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
юBы
#__inference__update_step_xla_283842gradientvariable"­
ІВЂ
FullArgSpec*
args"

jgradient

jvariable
jkey
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
юBы
#__inference__update_step_xla_283847gradientvariable"­
ІВЂ
FullArgSpec*
args"

jgradient

jvariable
jkey
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
юBы
#__inference__update_step_xla_283852gradientvariable"­
ІВЂ
FullArgSpec*
args"

jgradient

jvariable
jkey
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
0
Б0
В1"
trackable_list_wrapper
.
Џ	variables"
_generic_user_object
:  (2total
:  (2count
0
Е0
Ж1"
trackable_list_wrapper
.
Г	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapper
QbO
beta:0E__inference_conv1d_53_layer_call_and_return_conditional_losses_283911
TbR
	BiasAdd:0E__inference_conv1d_53_layer_call_and_return_conditional_losses_283911
QbO
beta:0E__inference_conv1d_53_layer_call_and_return_conditional_losses_283511
TbR
	BiasAdd:0E__inference_conv1d_53_layer_call_and_return_conditional_losses_283511
ZbX
conv1d_53/beta:0D__inference_model_57_layer_call_and_return_conditional_losses_283784
]b[
conv1d_53/BiasAdd:0D__inference_model_57_layer_call_and_return_conditional_losses_283784
ZbX
conv1d_53/beta:0D__inference_model_57_layer_call_and_return_conditional_losses_283832
]b[
conv1d_53/BiasAdd:0D__inference_model_57_layer_call_and_return_conditional_losses_283832
@b>
model_57/conv1d_53/beta:0!__inference__wrapped_model_283433
CbA
model_57/conv1d_53/BiasAdd:0!__inference__wrapped_model_283433В
K__inference_INJECTION_MASKS_layer_call_and_return_conditional_losses_283955cQR/Ђ,
%Ђ"
 
inputsџџџџџџџџџ
Њ ",Ђ)
"
tensor_0џџџџџџџџџ
 
0__inference_INJECTION_MASKS_layer_call_fn_283944XQR/Ђ,
%Ђ"
 
inputsџџџџџџџџџ
Њ "!
unknownџџџџџџџџџ
#__inference__update_step_xla_283837vpЂm
fЂc

gradient*
85	!Ђ
њ*

p
` VariableSpec 
`рИухцЭ?
Њ "
 
#__inference__update_step_xla_283842f`Ђ]
VЂS

gradient
0-	Ђ
њ

p
` VariableSpec 
`ряъхцЭ?
Њ "
 
#__inference__update_step_xla_283847nhЂe
^Ђ[

gradient
41	Ђ
њ

p
` VariableSpec 
`рщзчцЭ?
Њ "
 
#__inference__update_step_xla_283852f`Ђ]
VЂS

gradient
0-	Ђ
њ

p
` VariableSpec 
`рђцЭ?
Њ "
 №
!__inference__wrapped_model_283433Ъ89QRЂ|
uЂr
pЊm
6
	OFFSOURCE)&
	OFFSOURCEџџџџџџџџџ
3
ONSOURCE'$
ONSOURCEџџџџџџџџџ 
Њ "AЊ>
<
INJECTION_MASKS)&
injection_masksџџџџџџџџџД
E__inference_conv1d_53_layer_call_and_return_conditional_losses_283911k893Ђ0
)Ђ&
$!
inputsџџџџџџџџџ
Њ "0Ђ-
&#
tensor_0џџџџџџџџџ
 
*__inference_conv1d_53_layer_call_fn_283887`893Ђ0
)Ђ&
$!
inputsџџџџџџџџџ
Њ "%"
unknownџџџџџџџџџ­
F__inference_flatten_57_layer_call_and_return_conditional_losses_283935c3Ђ0
)Ђ&
$!
inputsџџџџџџџџџ
Њ ",Ђ)
"
tensor_0џџџџџџџџџ
 
+__inference_flatten_57_layer_call_fn_283929X3Ђ0
)Ђ&
$!
inputsџџџџџџџџџ
Њ "!
unknownџџџџџџџџџќ
#__inference_internal_grad_fn_284023дИЙЂ
|Ђy

 
,)
result_grads_0џџџџџџџџџ
,)
result_grads_1џџџџџџџџџ

result_grads_2 
Њ "B?

 
&#
tensor_1џџџџџџџџџ

tensor_2 ќ
#__inference_internal_grad_fn_284051дКЛЂ
|Ђy

 
,)
result_grads_0џџџџџџџџџ
,)
result_grads_1џџџџџџџџџ

result_grads_2 
Њ "B?

 
&#
tensor_1џџџџџџџџџ

tensor_2 ќ
#__inference_internal_grad_fn_284079дМНЂ
|Ђy

 
,)
result_grads_0џџџџџџџџџ
,)
result_grads_1џџџџџџџџџ

result_grads_2 
Њ "B?

 
&#
tensor_1џџџџџџџџџ

tensor_2 ќ
#__inference_internal_grad_fn_284107дОПЂ
|Ђy

 
,)
result_grads_0џџџџџџџџџ
,)
result_grads_1џџџџџџџџџ

result_grads_2 
Њ "B?

 
&#
tensor_1џџџџџџџџџ

tensor_2 ќ
#__inference_internal_grad_fn_284135дРСЂ
|Ђy

 
,)
result_grads_0џџџџџџџџџ
,)
result_grads_1џџџџџџџџџ

result_grads_2 
Њ "B?

 
&#
tensor_1џџџџџџџџџ

tensor_2 м
L__inference_max_pooling1d_69_layer_call_and_return_conditional_losses_283865EЂB
;Ђ8
63
inputs'џџџџџџџџџџџџџџџџџџџџџџџџџџџ
Њ "BЂ?
85
tensor_0'џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 Ж
1__inference_max_pooling1d_69_layer_call_fn_283857EЂB
;Ђ8
63
inputs'џџџџџџџџџџџџџџџџџџџџџџџџџџџ
Њ "74
unknown'џџџџџџџџџџџџџџџџџџџџџџџџџџџм
L__inference_max_pooling1d_70_layer_call_and_return_conditional_losses_283878EЂB
;Ђ8
63
inputs'џџџџџџџџџџџџџџџџџџџџџџџџџџџ
Њ "BЂ?
85
tensor_0'џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 Ж
1__inference_max_pooling1d_70_layer_call_fn_283870EЂB
;Ђ8
63
inputs'џџџџџџџџџџџџџџџџџџџџџџџџџџџ
Њ "74
unknown'џџџџџџџџџџџџџџџџџџџџџџџџџџџм
L__inference_max_pooling1d_71_layer_call_and_return_conditional_losses_283924EЂB
;Ђ8
63
inputs'џџџџџџџџџџџџџџџџџџџџџџџџџџџ
Њ "BЂ?
85
tensor_0'џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 Ж
1__inference_max_pooling1d_71_layer_call_fn_283916EЂB
;Ђ8
63
inputs'џџџџџџџџџџџџџџџџџџџџџџџџџџџ
Њ "74
unknown'џџџџџџџџџџџџџџџџџџџџџџџџџџџ
D__inference_model_57_layer_call_and_return_conditional_losses_283544П89QRЂ
}Ђz
pЊm
6
	OFFSOURCE)&
	OFFSOURCEџџџџџџџџџ
3
ONSOURCE'$
ONSOURCEџџџџџџџџџ 
p

 
Њ ",Ђ)
"
tensor_0џџџџџџџџџ
 
D__inference_model_57_layer_call_and_return_conditional_losses_283565П89QRЂ
}Ђz
pЊm
6
	OFFSOURCE)&
	OFFSOURCEџџџџџџџџџ
3
ONSOURCE'$
ONSOURCEџџџџџџџџџ 
p 

 
Њ ",Ђ)
"
tensor_0џџџџџџџџџ
 
D__inference_model_57_layer_call_and_return_conditional_losses_283784Я89QRЂ
Ђ
~Њ{
=
	OFFSOURCE0-
inputs_offsourceџџџџџџџџџ
:
ONSOURCE.+
inputs_onsourceџџџџџџџџџ 
p

 
Њ ",Ђ)
"
tensor_0џџџџџџџџџ
 
D__inference_model_57_layer_call_and_return_conditional_losses_283832Я89QRЂ
Ђ
~Њ{
=
	OFFSOURCE0-
inputs_offsourceџџџџџџџџџ
:
ONSOURCE.+
inputs_onsourceџџџџџџџџџ 
p 

 
Њ ",Ђ)
"
tensor_0џџџџџџџџџ
 т
)__inference_model_57_layer_call_fn_283601Д89QRЂ
}Ђz
pЊm
6
	OFFSOURCE)&
	OFFSOURCEџџџџџџџџџ
3
ONSOURCE'$
ONSOURCEџџџџџџџџџ 
p

 
Њ "!
unknownџџџџџџџџџт
)__inference_model_57_layer_call_fn_283636Д89QRЂ
}Ђz
pЊm
6
	OFFSOURCE)&
	OFFSOURCEџџџџџџџџџ
3
ONSOURCE'$
ONSOURCEџџџџџџџџџ 
p 

 
Њ "!
unknownџџџџџџџџџђ
)__inference_model_57_layer_call_fn_283722Ф89QRЂ
Ђ
~Њ{
=
	OFFSOURCE0-
inputs_offsourceџџџџџџџџџ
:
ONSOURCE.+
inputs_onsourceџџџџџџџџџ 
p

 
Њ "!
unknownџџџџџџџџџђ
)__inference_model_57_layer_call_fn_283736Ф89QRЂ
Ђ
~Њ{
=
	OFFSOURCE0-
inputs_offsourceџџџџџџџџџ
:
ONSOURCE.+
inputs_onsourceџџџџџџџџџ 
p 

 
Њ "!
unknownџџџџџџџџџА
C__inference_reshape_57_layer_call_and_return_conditional_losses_512i4Ђ1
*Ђ'
%"
inputsџџџџџџџџџ
Њ "1Ђ.
'$
tensor_0џџџџџџџџџ
 
(__inference_reshape_57_layer_call_fn_317^4Ђ1
*Ђ'
%"
inputsџџџџџџџџџ
Њ "&#
unknownџџџџџџџџџю
$__inference_signature_wrapper_283708Х89QRzЂw
Ђ 
pЊm
6
	OFFSOURCE)&
	offsourceџџџџџџџџџ
3
ONSOURCE'$
onsourceџџџџџџџџџ "AЊ>
<
INJECTION_MASKS)&
injection_masksџџџџџџџџџМ
N__inference_whiten_passthrough_28_layer_call_and_return_conditional_losses_767j5Ђ2
+Ђ(
&#
inputsџџџџџџџџџ
Њ "1Ђ.
'$
tensor_0џџџџџџџџџ
 
3__inference_whiten_passthrough_28_layer_call_fn_750_5Ђ2
+Ђ(
&#
inputsџџџџџџџџџ
Њ "&#
unknownџџџџџџџџџ