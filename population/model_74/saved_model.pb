Ћ
ЮЁ
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
$
DisableCopyOnRead
resource
.
Identity

input"T
output"T"	
Ttype
u
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:
2	

MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool("
allow_missing_filesbool( 
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
 "serve*2.12.12v2.12.0-25-g8e2b6655c0c8Хд
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
w
Adam/v/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*
shared_nameAdam/v/kernel
p
!Adam/v/kernel/Read/ReadVariableOpReadVariableOpAdam/v/kernel*
_output_shapes
:	*
dtype0
w
Adam/m/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*
shared_nameAdam/m/kernel
p
!Adam/m/kernel/Read/ReadVariableOpReadVariableOpAdam/m/kernel*
_output_shapes
:	*
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
i
kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*
shared_namekernel
b
kernel/Read/ReadVariableOpReadVariableOpkernel*
_output_shapes
:	*
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
№
StatefulPartitionedCallStatefulPartitionedCallserving_default_OFFSOURCEserving_default_ONSOURCEkernelbias*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*<
config_proto,*

CPU

GPU(2*0J

  zE8 *-
f(R&
$__inference_signature_wrapper_282239

NoOpNoOp
+
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*ж*
valueЬ*BЩ* BТ*
ц
layer-0
layer-1
layer-2
layer-3
layer-4
layer-5
layer_with_weights-0
layer-6
	variables
	trainable_variables

regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
_default_save_signature
	optimizer

signatures
#_self_saveable_object_factories*
'
#_self_saveable_object_factories* 
Г
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
#_self_saveable_object_factories* 
Г
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
# _self_saveable_object_factories* 
Ъ
!	variables
"trainable_variables
#regularization_losses
$	keras_api
%__call__
*&&call_and_return_all_conditional_losses
'_random_generator
#(_self_saveable_object_factories* 
Г
)	variables
*trainable_variables
+regularization_losses
,	keras_api
-__call__
*.&call_and_return_all_conditional_losses
#/_self_saveable_object_factories* 
'
#0_self_saveable_object_factories* 
Ы
1	variables
2trainable_variables
3regularization_losses
4	keras_api
5__call__
*6&call_and_return_all_conditional_losses

7kernel
8bias
#9_self_saveable_object_factories*

70
81*

70
81*
* 
А
:non_trainable_variables

;layers
<metrics
=layer_regularization_losses
>layer_metrics
	variables
	trainable_variables

regularization_losses
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*
6
?trace_0
@trace_1
Atrace_2
Btrace_3* 
6
Ctrace_0
Dtrace_1
Etrace_2
Ftrace_3* 
* 

G
_variables
H_iterations
I_learning_rate
J_index_dict
K
_momentums
L_velocities
M_update_step_xla*

Nserving_default* 
* 
* 
* 
* 
* 

Onon_trainable_variables

Players
Qmetrics
Rlayer_regularization_losses
Slayer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses* 

Ttrace_0* 

Utrace_0* 
* 
* 
* 
* 

Vnon_trainable_variables

Wlayers
Xmetrics
Ylayer_regularization_losses
Zlayer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses* 

[trace_0* 

\trace_0* 
* 
* 
* 
* 

]non_trainable_variables

^layers
_metrics
`layer_regularization_losses
alayer_metrics
!	variables
"trainable_variables
#regularization_losses
%__call__
*&&call_and_return_all_conditional_losses
&&"call_and_return_conditional_losses* 

btrace_0
ctrace_1* 

dtrace_0
etrace_1* 
'
#f_self_saveable_object_factories* 
* 
* 
* 
* 

gnon_trainable_variables

hlayers
imetrics
jlayer_regularization_losses
klayer_metrics
)	variables
*trainable_variables
+regularization_losses
-__call__
*.&call_and_return_all_conditional_losses
&."call_and_return_conditional_losses* 

ltrace_0* 

mtrace_0* 
* 
* 

70
81*

70
81*
* 

nnon_trainable_variables

olayers
pmetrics
qlayer_regularization_losses
rlayer_metrics
1	variables
2trainable_variables
3regularization_losses
5__call__
*6&call_and_return_all_conditional_losses
&6"call_and_return_conditional_losses*

strace_0* 

ttrace_0* 
VP
VARIABLE_VALUEkernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE*
RL
VARIABLE_VALUEbias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
5
0
1
2
3
4
5
6*

u0
v1*
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
'
H0
w1
x2
y3
z4*
SM
VARIABLE_VALUE	iteration0optimizer/_iterations/.ATTRIBUTES/VARIABLE_VALUE*
ZT
VARIABLE_VALUElearning_rate3optimizer/_learning_rate/.ATTRIBUTES/VARIABLE_VALUE*
* 

w0
y1*

x0
z1*

{trace_0
|trace_1* 
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
9
}	variables
~	keras_api
	total

count*
M
	variables
	keras_api

total

count

_fn_kwargs*
XR
VARIABLE_VALUEAdam/m/kernel1optimizer/_variables/1/.ATTRIBUTES/VARIABLE_VALUE*
XR
VARIABLE_VALUEAdam/v/kernel1optimizer/_variables/2/.ATTRIBUTES/VARIABLE_VALUE*
VP
VARIABLE_VALUEAdam/m/bias1optimizer/_variables/3/.ATTRIBUTES/VARIABLE_VALUE*
VP
VARIABLE_VALUEAdam/v/bias1optimizer/_variables/4/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 

0
1*

}	variables*
UO
VARIABLE_VALUEtotal_14keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE*
UO
VARIABLE_VALUEcount_14keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE*

0
1*

	variables*
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
Ж
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filenamekernelbias	iterationlearning_rateAdam/m/kernelAdam/v/kernelAdam/m/biasAdam/v/biastotal_1count_1totalcountConst*
Tin
2*
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
__inference__traced_save_282464
Б
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamekernelbias	iterationlearning_rateAdam/m/kernelAdam/v/kernelAdam/m/biasAdam/v/biastotal_1count_1totalcount*
Tin
2*
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
"__inference__traced_restore_282510 
Г
Ш
!__inference__wrapped_model_282063
	offsource
onsourceJ
7model_74_injection_masks_matmul_readvariableop_resource:	F
8model_74_injection_masks_biasadd_readvariableop_resource:
identityЂ/model_74/INJECTION_MASKS/BiasAdd/ReadVariableOpЂ.model_74/INJECTION_MASKS/MatMul/ReadVariableOpЪ
.model_74/whiten_passthrough_37/PartitionedCallPartitionedCall	offsource*
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
)__inference_restored_function_body_282044э
#model_74/reshape_74/PartitionedCallPartitionedCall7model_74/whiten_passthrough_37/PartitionedCall:output:0*
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
)__inference_restored_function_body_282050
model_74/dropout_81/IdentityIdentity,model_74/reshape_74/PartitionedCall:output:0*
T0*,
_output_shapes
:џџџџџџџџџj
model_74/flatten_74/ConstConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   Є
model_74/flatten_74/ReshapeReshape%model_74/dropout_81/Identity:output:0"model_74/flatten_74/Const:output:0*
T0*(
_output_shapes
:џџџџџџџџџЇ
.model_74/INJECTION_MASKS/MatMul/ReadVariableOpReadVariableOp7model_74_injection_masks_matmul_readvariableop_resource*
_output_shapes
:	*
dtype0Й
model_74/INJECTION_MASKS/MatMulMatMul$model_74/flatten_74/Reshape:output:06model_74/INJECTION_MASKS/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџЄ
/model_74/INJECTION_MASKS/BiasAdd/ReadVariableOpReadVariableOp8model_74_injection_masks_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0С
 model_74/INJECTION_MASKS/BiasAddBiasAdd)model_74/INJECTION_MASKS/MatMul:product:07model_74/INJECTION_MASKS/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ
 model_74/INJECTION_MASKS/SigmoidSigmoid)model_74/INJECTION_MASKS/BiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџs
IdentityIdentity$model_74/INJECTION_MASKS/Sigmoid:y:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџЉ
NoOpNoOp0^model_74/INJECTION_MASKS/BiasAdd/ReadVariableOp/^model_74/INJECTION_MASKS/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:џџџџџџџџџ:џџџџџџџџџ : : 2b
/model_74/INJECTION_MASKS/BiasAdd/ReadVariableOp/model_74/INJECTION_MASKS/BiasAdd/ReadVariableOp2`
.model_74/INJECTION_MASKS/MatMul/ReadVariableOp.model_74/INJECTION_MASKS/MatMul/ReadVariableOp:VR
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
ф

D__inference_model_74_layer_call_and_return_conditional_losses_282110
	offsource
onsource)
injection_masks_282104:	$
injection_masks_282106:
identityЂ'INJECTION_MASKS/StatefulPartitionedCallЂ"dropout_81/StatefulPartitionedCallС
%whiten_passthrough_37/PartitionedCallPartitionedCall	offsource*
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
)__inference_restored_function_body_282044л
reshape_74/PartitionedCallPartitionedCall.whiten_passthrough_37/PartitionedCall:output:0*
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
)__inference_restored_function_body_282050§
"dropout_81/StatefulPartitionedCallStatefulPartitionedCall#reshape_74/PartitionedCall:output:0*
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
  zE8 *O
fJRH
F__inference_dropout_81_layer_call_and_return_conditional_losses_282082ё
flatten_74/PartitionedCallPartitionedCall+dropout_81/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:џџџџџџџџџ* 
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
F__inference_flatten_74_layer_call_and_return_conditional_losses_282090И
'INJECTION_MASKS/StatefulPartitionedCallStatefulPartitionedCall#flatten_74/PartitionedCall:output:0injection_masks_282104injection_masks_282106*
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
K__inference_INJECTION_MASKS_layer_call_and_return_conditional_losses_282103
IdentityIdentity0INJECTION_MASKS/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ
NoOpNoOp(^INJECTION_MASKS/StatefulPartitionedCall#^dropout_81/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:џџџџџџџџџ:џџџџџџџџџ : : 2R
'INJECTION_MASKS/StatefulPartitionedCall'INJECTION_MASKS/StatefulPartitionedCall2H
"dropout_81/StatefulPartitionedCall"dropout_81/StatefulPartitionedCall:VR
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
Ъ

e
F__inference_dropout_81_layer_call_and_return_conditional_losses_282082

inputs
identityR
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *с\@i
dropout/MulMulinputsdropout/Const:output:0*
T0*,
_output_shapes
:џџџџџџџџџQ
dropout/ShapeShapeinputs*
T0*
_output_shapes
::эЯ
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*,
_output_shapes
:џџџџџџџџџ*
dtype0*
seedш[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *жB?Ћ
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*,
_output_shapes
:џџџџџџџџџT
dropout/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    
dropout/SelectV2SelectV2dropout/GreaterEqual:z:0dropout/Mul:z:0dropout/Const_1:output:0*
T0*,
_output_shapes
:џџџџџџџџџf
IdentityIdentitydropout/SelectV2:output:0*
T0*,
_output_shapes
:џџџџџџџџџ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:џџџџџџџџџ:T P
,
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
ж
O
3__inference_whiten_passthrough_37_layer_call_fn_381

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
N__inference_whiten_passthrough_37_layer_call_and_return_conditional_losses_376e
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
т

0__inference_INJECTION_MASKS_layer_call_fn_282357

inputs
unknown:	
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
K__inference_INJECTION_MASKS_layer_call_and_return_conditional_losses_282103o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:џџџџџџџџџ: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs

d
+__inference_dropout_81_layer_call_fn_282315

inputs
identityЂStatefulPartitionedCallе
StatefulPartitionedCallStatefulPartitionedCallinputs*
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
  zE8 *O
fJRH
F__inference_dropout_81_layer_call_and_return_conditional_losses_282082t
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*,
_output_shapes
:џџџџџџџџџ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:џџџџџџџџџ22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
Ў
ь
D__inference_model_74_layer_call_and_return_conditional_losses_282171
inputs_1

inputs)
injection_masks_282165:	$
injection_masks_282167:
identityЂ'INJECTION_MASKS/StatefulPartitionedCallР
%whiten_passthrough_37/PartitionedCallPartitionedCallinputs_1*
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
)__inference_restored_function_body_282044л
reshape_74/PartitionedCallPartitionedCall.whiten_passthrough_37/PartitionedCall:output:0*
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
)__inference_restored_function_body_282050э
dropout_81/PartitionedCallPartitionedCall#reshape_74/PartitionedCall:output:0*
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
  zE8 *O
fJRH
F__inference_dropout_81_layer_call_and_return_conditional_losses_282120щ
flatten_74/PartitionedCallPartitionedCall#dropout_81/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:џџџџџџџџџ* 
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
F__inference_flatten_74_layer_call_and_return_conditional_losses_282090И
'INJECTION_MASKS/StatefulPartitionedCallStatefulPartitionedCall#flatten_74/PartitionedCall:output:0injection_masks_282165injection_masks_282167*
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
K__inference_INJECTION_MASKS_layer_call_and_return_conditional_losses_282103
IdentityIdentity0INJECTION_MASKS/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџp
NoOpNoOp(^INJECTION_MASKS/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:џџџџџџџџџ:џџџџџџџџџ : : 2R
'INJECTION_MASKS/StatefulPartitionedCall'INJECTION_MASKS/StatefulPartitionedCall:TP
,
_output_shapes
:џџџџџџџџџ 
 
_user_specified_nameinputs:U Q
-
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
Л
P
#__inference__update_step_xla_282305
gradient
variable:	*
_XlaMustCompile(*(
_construction_contextkEagerRuntime* 
_input_shapes
:	: *
	_noinline(:($
"
_user_specified_name
variable:I E

_output_shapes
:	
"
_user_specified_name
gradient
 
E
)__inference_restored_function_body_282050

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
C__inference_reshape_74_layer_call_and_return_conditional_losses_280e
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
	
Ж
)__inference_model_74_layer_call_fn_282249
inputs_offsource
inputs_onsource
unknown:	
	unknown_0:
identityЂStatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputs_offsourceinputs_onsourceunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*<
config_proto,*

CPU

GPU(2*0J

  zE8 *M
fHRF
D__inference_model_74_layer_call_and_return_conditional_losses_282147o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:џџџџџџџџџ:џџџџџџџџџ : : 22
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
	
Ж
)__inference_model_74_layer_call_fn_282259
inputs_offsource
inputs_onsource
unknown:	
	unknown_0:
identityЂStatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputs_offsourceinputs_onsourceunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*<
config_proto,*

CPU

GPU(2*0J

  zE8 *M
fHRF
D__inference_model_74_layer_call_and_return_conditional_losses_282171o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:џџџџџџџџџ:џџџџџџџџџ : : 22
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
№
Ј
)__inference_model_74_layer_call_fn_282178
	offsource
onsource
unknown:	
	unknown_0:
identityЂStatefulPartitionedCallі
StatefulPartitionedCallStatefulPartitionedCall	offsourceonsourceunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*<
config_proto,*

CPU

GPU(2*0J

  zE8 *M
fHRF
D__inference_model_74_layer_call_and_return_conditional_losses_282171o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:џџџџџџџџџ:џџџџџџџџџ : : 22
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
Ш
Ѓ
$__inference_signature_wrapper_282239
	offsource
onsource
unknown:	
	unknown_0:
identityЂStatefulPartitionedCallг
StatefulPartitionedCallStatefulPartitionedCall	offsourceonsourceunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*<
config_proto,*

CPU

GPU(2*0J

  zE8 **
f%R#
!__inference__wrapped_model_282063o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:џџџџџџџџџ:џџџџџџџџџ : : 22
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
Ќ
K
#__inference__update_step_xla_282310
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
о
_
C__inference_reshape_74_layer_call_and_return_conditional_losses_578

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
О
D
(__inference_reshape_74_layer_call_fn_583

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
C__inference_reshape_74_layer_call_and_return_conditional_losses_578e
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
Т
b
F__inference_flatten_74_layer_call_and_return_conditional_losses_282348

inputs
identityV
ConstConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   ]
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:џџџџџџџџџY
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:џџџџџџџџџ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:џџџџџџџџџ:T P
,
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
Ћ^


__inference__traced_save_282464
file_prefix0
read_disablecopyonread_kernel:	+
read_1_disablecopyonread_bias:,
"read_2_disablecopyonread_iteration:	 0
&read_3_disablecopyonread_learning_rate: 9
&read_4_disablecopyonread_adam_m_kernel:	9
&read_5_disablecopyonread_adam_v_kernel:	2
$read_6_disablecopyonread_adam_m_bias:2
$read_7_disablecopyonread_adam_v_bias:*
 read_8_disablecopyonread_total_1: *
 read_9_disablecopyonread_count_1: )
read_10_disablecopyonread_total: )
read_11_disablecopyonread_count: 
savev2_const
identity_25ЂMergeV2CheckpointsЂRead/DisableCopyOnReadЂRead/ReadVariableOpЂRead_1/DisableCopyOnReadЂRead_1/ReadVariableOpЂRead_10/DisableCopyOnReadЂRead_10/ReadVariableOpЂRead_11/DisableCopyOnReadЂRead_11/ReadVariableOpЂRead_2/DisableCopyOnReadЂRead_2/ReadVariableOpЂRead_3/DisableCopyOnReadЂRead_3/ReadVariableOpЂRead_4/DisableCopyOnReadЂRead_4/ReadVariableOpЂRead_5/DisableCopyOnReadЂRead_5/ReadVariableOpЂRead_6/DisableCopyOnReadЂRead_6/ReadVariableOpЂRead_7/DisableCopyOnReadЂRead_7/ReadVariableOpЂRead_8/DisableCopyOnReadЂRead_8/ReadVariableOpЂRead_9/DisableCopyOnReadЂRead_9/ReadVariableOpw
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
: o
Read/DisableCopyOnReadDisableCopyOnReadread_disablecopyonread_kernel"/device:CPU:0*
_output_shapes
 
Read/ReadVariableOpReadVariableOpread_disablecopyonread_kernel^Read/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:	*
dtype0j
IdentityIdentityRead/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:	b

Identity_1IdentityIdentity:output:0"/device:CPU:0*
T0*
_output_shapes
:	q
Read_1/DisableCopyOnReadDisableCopyOnReadread_1_disablecopyonread_bias"/device:CPU:0*
_output_shapes
 
Read_1/ReadVariableOpReadVariableOpread_1_disablecopyonread_bias^Read_1/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0i

Identity_2IdentityRead_1/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:_

Identity_3IdentityIdentity_2:output:0"/device:CPU:0*
T0*
_output_shapes
:v
Read_2/DisableCopyOnReadDisableCopyOnRead"read_2_disablecopyonread_iteration"/device:CPU:0*
_output_shapes
 
Read_2/ReadVariableOpReadVariableOp"read_2_disablecopyonread_iteration^Read_2/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0	e

Identity_4IdentityRead_2/ReadVariableOp:value:0"/device:CPU:0*
T0	*
_output_shapes
: [

Identity_5IdentityIdentity_4:output:0"/device:CPU:0*
T0	*
_output_shapes
: z
Read_3/DisableCopyOnReadDisableCopyOnRead&read_3_disablecopyonread_learning_rate"/device:CPU:0*
_output_shapes
 
Read_3/ReadVariableOpReadVariableOp&read_3_disablecopyonread_learning_rate^Read_3/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0e

Identity_6IdentityRead_3/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: [

Identity_7IdentityIdentity_6:output:0"/device:CPU:0*
T0*
_output_shapes
: z
Read_4/DisableCopyOnReadDisableCopyOnRead&read_4_disablecopyonread_adam_m_kernel"/device:CPU:0*
_output_shapes
 Ї
Read_4/ReadVariableOpReadVariableOp&read_4_disablecopyonread_adam_m_kernel^Read_4/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:	*
dtype0n

Identity_8IdentityRead_4/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:	d

Identity_9IdentityIdentity_8:output:0"/device:CPU:0*
T0*
_output_shapes
:	z
Read_5/DisableCopyOnReadDisableCopyOnRead&read_5_disablecopyonread_adam_v_kernel"/device:CPU:0*
_output_shapes
 Ї
Read_5/ReadVariableOpReadVariableOp&read_5_disablecopyonread_adam_v_kernel^Read_5/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:	*
dtype0o
Identity_10IdentityRead_5/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:	f
Identity_11IdentityIdentity_10:output:0"/device:CPU:0*
T0*
_output_shapes
:	x
Read_6/DisableCopyOnReadDisableCopyOnRead$read_6_disablecopyonread_adam_m_bias"/device:CPU:0*
_output_shapes
  
Read_6/ReadVariableOpReadVariableOp$read_6_disablecopyonread_adam_m_bias^Read_6/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0j
Identity_12IdentityRead_6/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:a
Identity_13IdentityIdentity_12:output:0"/device:CPU:0*
T0*
_output_shapes
:x
Read_7/DisableCopyOnReadDisableCopyOnRead$read_7_disablecopyonread_adam_v_bias"/device:CPU:0*
_output_shapes
  
Read_7/ReadVariableOpReadVariableOp$read_7_disablecopyonread_adam_v_bias^Read_7/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0j
Identity_14IdentityRead_7/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:a
Identity_15IdentityIdentity_14:output:0"/device:CPU:0*
T0*
_output_shapes
:t
Read_8/DisableCopyOnReadDisableCopyOnRead read_8_disablecopyonread_total_1"/device:CPU:0*
_output_shapes
 
Read_8/ReadVariableOpReadVariableOp read_8_disablecopyonread_total_1^Read_8/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0f
Identity_16IdentityRead_8/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: ]
Identity_17IdentityIdentity_16:output:0"/device:CPU:0*
T0*
_output_shapes
: t
Read_9/DisableCopyOnReadDisableCopyOnRead read_9_disablecopyonread_count_1"/device:CPU:0*
_output_shapes
 
Read_9/ReadVariableOpReadVariableOp read_9_disablecopyonread_count_1^Read_9/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0f
Identity_18IdentityRead_9/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: ]
Identity_19IdentityIdentity_18:output:0"/device:CPU:0*
T0*
_output_shapes
: t
Read_10/DisableCopyOnReadDisableCopyOnReadread_10_disablecopyonread_total"/device:CPU:0*
_output_shapes
 
Read_10/ReadVariableOpReadVariableOpread_10_disablecopyonread_total^Read_10/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0g
Identity_20IdentityRead_10/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: ]
Identity_21IdentityIdentity_20:output:0"/device:CPU:0*
T0*
_output_shapes
: t
Read_11/DisableCopyOnReadDisableCopyOnReadread_11_disablecopyonread_count"/device:CPU:0*
_output_shapes
 
Read_11/ReadVariableOpReadVariableOpread_11_disablecopyonread_count^Read_11/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0g
Identity_22IdentityRead_11/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: ]
Identity_23IdentityIdentity_22:output:0"/device:CPU:0*
T0*
_output_shapes
: 
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*Ќ
valueЂBB6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB0optimizer/_iterations/.ATTRIBUTES/VARIABLE_VALUEB3optimizer/_learning_rate/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/1/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/2/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/3/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/4/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*-
value$B"B B B B B B B B B B B B B ч
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0Identity_1:output:0Identity_3:output:0Identity_5:output:0Identity_7:output:0Identity_9:output:0Identity_11:output:0Identity_13:output:0Identity_15:output:0Identity_17:output:0Identity_19:output:0Identity_21:output:0Identity_23:output:0savev2_const"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtypes
2	
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:Г
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 i
Identity_24Identityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: U
Identity_25IdentityIdentity_24:output:0^NoOp*
T0*
_output_shapes
: П
NoOpNoOp^MergeV2Checkpoints^Read/DisableCopyOnRead^Read/ReadVariableOp^Read_1/DisableCopyOnRead^Read_1/ReadVariableOp^Read_10/DisableCopyOnRead^Read_10/ReadVariableOp^Read_11/DisableCopyOnRead^Read_11/ReadVariableOp^Read_2/DisableCopyOnRead^Read_2/ReadVariableOp^Read_3/DisableCopyOnRead^Read_3/ReadVariableOp^Read_4/DisableCopyOnRead^Read_4/ReadVariableOp^Read_5/DisableCopyOnRead^Read_5/ReadVariableOp^Read_6/DisableCopyOnRead^Read_6/ReadVariableOp^Read_7/DisableCopyOnRead^Read_7/ReadVariableOp^Read_8/DisableCopyOnRead^Read_8/ReadVariableOp^Read_9/DisableCopyOnRead^Read_9/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "#
identity_25Identity_25:output:0*/
_input_shapes
: : : : : : : : : : : : : : 2(
MergeV2CheckpointsMergeV2Checkpoints20
Read/DisableCopyOnReadRead/DisableCopyOnRead2*
Read/ReadVariableOpRead/ReadVariableOp24
Read_1/DisableCopyOnReadRead_1/DisableCopyOnRead2.
Read_1/ReadVariableOpRead_1/ReadVariableOp26
Read_10/DisableCopyOnReadRead_10/DisableCopyOnRead20
Read_10/ReadVariableOpRead_10/ReadVariableOp26
Read_11/DisableCopyOnReadRead_11/DisableCopyOnRead20
Read_11/ReadVariableOpRead_11/ReadVariableOp24
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
Read_9/ReadVariableOpRead_9/ReadVariableOp:

_output_shapes
: :C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
Ш
B
__inference_crop_samples_342
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
Ѕ

§
K__inference_INJECTION_MASKS_layer_call_and_return_conditional_losses_282103

inputs1
matmul_readvariableop_resource:	-
biasadd_readvariableop_resource:
identityЂBiasAdd/ReadVariableOpЂMatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	*
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
_construction_contextkEagerRuntime*+
_input_shapes
:џџџџџџџџџ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
э
d
F__inference_dropout_81_layer_call_and_return_conditional_losses_282120

inputs

identity_1S
IdentityIdentityinputs*
T0*,
_output_shapes
:џџџџџџџџџ`

Identity_1IdentityIdentity:output:0*
T0*,
_output_shapes
:џџџџџџџџџ"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:џџџџџџџџџ:T P
,
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
З
я
D__inference_model_74_layer_call_and_return_conditional_losses_282129
	offsource
onsource)
injection_masks_282123:	$
injection_masks_282125:
identityЂ'INJECTION_MASKS/StatefulPartitionedCallС
%whiten_passthrough_37/PartitionedCallPartitionedCall	offsource*
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
)__inference_restored_function_body_282044л
reshape_74/PartitionedCallPartitionedCall.whiten_passthrough_37/PartitionedCall:output:0*
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
)__inference_restored_function_body_282050э
dropout_81/PartitionedCallPartitionedCall#reshape_74/PartitionedCall:output:0*
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
  zE8 *O
fJRH
F__inference_dropout_81_layer_call_and_return_conditional_losses_282120щ
flatten_74/PartitionedCallPartitionedCall#dropout_81/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:џџџџџџџџџ* 
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
F__inference_flatten_74_layer_call_and_return_conditional_losses_282090И
'INJECTION_MASKS/StatefulPartitionedCallStatefulPartitionedCall#flatten_74/PartitionedCall:output:0injection_masks_282123injection_masks_282125*
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
K__inference_INJECTION_MASKS_layer_call_and_return_conditional_losses_282103
IdentityIdentity0INJECTION_MASKS/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџp
NoOpNoOp(^INJECTION_MASKS/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:џџџџџџџџџ:џџџџџџџџџ : : 2R
'INJECTION_MASKS/StatefulPartitionedCall'INJECTION_MASKS/StatefulPartitionedCall:VR
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
э
d
F__inference_dropout_81_layer_call_and_return_conditional_losses_282337

inputs

identity_1S
IdentityIdentityinputs*
T0*,
_output_shapes
:џџџџџџџџџ`

Identity_1IdentityIdentity:output:0*
T0*,
_output_shapes
:џџџџџџџџџ"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:џџџџџџџџџ:T P
,
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
Ѓ5
Ш
"__inference__traced_restore_282510
file_prefix*
assignvariableop_kernel:	%
assignvariableop_1_bias:&
assignvariableop_2_iteration:	 *
 assignvariableop_3_learning_rate: 3
 assignvariableop_4_adam_m_kernel:	3
 assignvariableop_5_adam_v_kernel:	,
assignvariableop_6_adam_m_bias:,
assignvariableop_7_adam_v_bias:$
assignvariableop_8_total_1: $
assignvariableop_9_count_1: #
assignvariableop_10_total: #
assignvariableop_11_count: 
identity_13ЂAssignVariableOpЂAssignVariableOp_1ЂAssignVariableOp_10ЂAssignVariableOp_11ЂAssignVariableOp_2ЂAssignVariableOp_3ЂAssignVariableOp_4ЂAssignVariableOp_5ЂAssignVariableOp_6ЂAssignVariableOp_7ЂAssignVariableOp_8ЂAssignVariableOp_9
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*Ќ
valueЂBB6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB0optimizer/_iterations/.ATTRIBUTES/VARIABLE_VALUEB3optimizer/_learning_rate/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/1/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/2/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/3/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/4/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*-
value$B"B B B B B B B B B B B B B п
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*H
_output_shapes6
4:::::::::::::*
dtypes
2	[
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:Њ
AssignVariableOpAssignVariableOpassignvariableop_kernelIdentity:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:Ў
AssignVariableOp_1AssignVariableOpassignvariableop_1_biasIdentity_1:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0	*
_output_shapes
:Г
AssignVariableOp_2AssignVariableOpassignvariableop_2_iterationIdentity_2:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0	]

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:З
AssignVariableOp_3AssignVariableOp assignvariableop_3_learning_rateIdentity_3:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:З
AssignVariableOp_4AssignVariableOp assignvariableop_4_adam_m_kernelIdentity_4:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:З
AssignVariableOp_5AssignVariableOp assignvariableop_5_adam_v_kernelIdentity_5:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:Е
AssignVariableOp_6AssignVariableOpassignvariableop_6_adam_m_biasIdentity_6:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:Е
AssignVariableOp_7AssignVariableOpassignvariableop_7_adam_v_biasIdentity_7:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:Б
AssignVariableOp_8AssignVariableOpassignvariableop_8_total_1Identity_8:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:Б
AssignVariableOp_9AssignVariableOpassignvariableop_9_count_1Identity_9:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:В
AssignVariableOp_10AssignVariableOpassignvariableop_10_totalIdentity_10:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:В
AssignVariableOp_11AssignVariableOpassignvariableop_11_countIdentity_11:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0Y
NoOpNoOp"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 з
Identity_12Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: W
Identity_13IdentityIdentity_12:output:0^NoOp_1*
T0*
_output_shapes
: Ф
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*"
_acd_function_control_output(*
_output_shapes
 "#
identity_13Identity_13:output:0*-
_input_shapes
: : : : : : : : : : : : : 2*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112(
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
№
Ј
)__inference_model_74_layer_call_fn_282154
	offsource
onsource
unknown:	
	unknown_0:
identityЂStatefulPartitionedCallі
StatefulPartitionedCallStatefulPartitionedCall	offsourceonsourceunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*<
config_proto,*

CPU

GPU(2*0J

  zE8 *M
fHRF
D__inference_model_74_layer_call_and_return_conditional_losses_282147o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:џџџџџџџџџ:џџџџџџџџџ : : 22
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
Т
b
F__inference_flatten_74_layer_call_and_return_conditional_losses_282090

inputs
identityV
ConstConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   ]
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:џџџџџџџџџY
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:џџџџџџџџџ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:џџџџџџџџџ:T P
,
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
Ъ

e
F__inference_dropout_81_layer_call_and_return_conditional_losses_282332

inputs
identityR
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *с\@i
dropout/MulMulinputsdropout/Const:output:0*
T0*,
_output_shapes
:џџџџџџџџџQ
dropout/ShapeShapeinputs*
T0*
_output_shapes
::эЯ
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*,
_output_shapes
:џџџџџџџџџ*
dtype0*
seedш[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *жB?Ћ
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*,
_output_shapes
:џџџџџџџџџT
dropout/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    
dropout/SelectV2SelectV2dropout/GreaterEqual:z:0dropout/Mul:z:0dropout/Const_1:output:0*
T0*,
_output_shapes
:џџџџџџџџџf
IdentityIdentitydropout/SelectV2:output:0*
T0*,
_output_shapes
:џџџџџџџџџ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:џџџџџџџџџ:T P
,
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
Ф
G
+__inference_dropout_81_layer_call_fn_282320

inputs
identityХ
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
  zE8 *O
fJRH
F__inference_dropout_81_layer_call_and_return_conditional_losses_282120e
IdentityIdentityPartitionedCall:output:0*
T0*,
_output_shapes
:џџџџџџџџџ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:џџџџџџџџџ:T P
,
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
Ѕ

§
K__inference_INJECTION_MASKS_layer_call_and_return_conditional_losses_282368

inputs1
matmul_readvariableop_resource:	-
biasadd_readvariableop_resource:
identityЂBiasAdd/ReadVariableOpЂMatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	*
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
_construction_contextkEagerRuntime*+
_input_shapes
:џџџџџџџџџ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
о
_
C__inference_reshape_74_layer_call_and_return_conditional_losses_280

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
з
е
D__inference_model_74_layer_call_and_return_conditional_losses_282300
inputs_offsource
inputs_onsourceA
.injection_masks_matmul_readvariableop_resource:	=
/injection_masks_biasadd_readvariableop_resource:
identityЂ&INJECTION_MASKS/BiasAdd/ReadVariableOpЂ%INJECTION_MASKS/MatMul/ReadVariableOpШ
%whiten_passthrough_37/PartitionedCallPartitionedCallinputs_offsource*
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
)__inference_restored_function_body_282044л
reshape_74/PartitionedCallPartitionedCall.whiten_passthrough_37/PartitionedCall:output:0*
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
)__inference_restored_function_body_282050{
dropout_81/IdentityIdentity#reshape_74/PartitionedCall:output:0*
T0*,
_output_shapes
:џџџџџџџџџa
flatten_74/ConstConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   
flatten_74/ReshapeReshapedropout_81/Identity:output:0flatten_74/Const:output:0*
T0*(
_output_shapes
:џџџџџџџџџ
%INJECTION_MASKS/MatMul/ReadVariableOpReadVariableOp.injection_masks_matmul_readvariableop_resource*
_output_shapes
:	*
dtype0
INJECTION_MASKS/MatMulMatMulflatten_74/Reshape:output:0-INJECTION_MASKS/MatMul/ReadVariableOp:value:0*
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
:џџџџџџџџџ
NoOpNoOp'^INJECTION_MASKS/BiasAdd/ReadVariableOp&^INJECTION_MASKS/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:џџџџџџџџџ:џџџџџџџџџ : : 2P
&INJECTION_MASKS/BiasAdd/ReadVariableOp&INJECTION_MASKS/BiasAdd/ReadVariableOp2N
%INJECTION_MASKS/MatMul/ReadVariableOp%INJECTION_MASKS/MatMul/ReadVariableOp:]Y
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
є
j
N__inference_whiten_passthrough_37_layer_call_and_return_conditional_losses_376

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
__inference_crop_samples_342I
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
­
E
)__inference_restored_function_body_282044

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
N__inference_whiten_passthrough_37_layer_call_and_return_conditional_losses_484e
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
є
j
N__inference_whiten_passthrough_37_layer_call_and_return_conditional_losses_484

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
__inference_crop_samples_342I
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
М
G
+__inference_flatten_74_layer_call_fn_282342

inputs
identityС
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:џџџџџџџџџ* 
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
F__inference_flatten_74_layer_call_and_return_conditional_losses_282090a
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:џџџџџџџџџ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:џџџџџџџџџ:T P
,
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
ё
е
D__inference_model_74_layer_call_and_return_conditional_losses_282283
inputs_offsource
inputs_onsourceA
.injection_masks_matmul_readvariableop_resource:	=
/injection_masks_biasadd_readvariableop_resource:
identityЂ&INJECTION_MASKS/BiasAdd/ReadVariableOpЂ%INJECTION_MASKS/MatMul/ReadVariableOpШ
%whiten_passthrough_37/PartitionedCallPartitionedCallinputs_offsource*
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
)__inference_restored_function_body_282044л
reshape_74/PartitionedCallPartitionedCall.whiten_passthrough_37/PartitionedCall:output:0*
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
)__inference_restored_function_body_282050]
dropout_81/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *с\@
dropout_81/dropout/MulMul#reshape_74/PartitionedCall:output:0!dropout_81/dropout/Const:output:0*
T0*,
_output_shapes
:џџџџџџџџџy
dropout_81/dropout/ShapeShape#reshape_74/PartitionedCall:output:0*
T0*
_output_shapes
::эЯД
/dropout_81/dropout/random_uniform/RandomUniformRandomUniform!dropout_81/dropout/Shape:output:0*
T0*,
_output_shapes
:џџџџџџџџџ*
dtype0*
seedшf
!dropout_81/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *жB?Ь
dropout_81/dropout/GreaterEqualGreaterEqual8dropout_81/dropout/random_uniform/RandomUniform:output:0*dropout_81/dropout/GreaterEqual/y:output:0*
T0*,
_output_shapes
:џџџџџџџџџ_
dropout_81/dropout/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    Ф
dropout_81/dropout/SelectV2SelectV2#dropout_81/dropout/GreaterEqual:z:0dropout_81/dropout/Mul:z:0#dropout_81/dropout/Const_1:output:0*
T0*,
_output_shapes
:џџџџџџџџџa
flatten_74/ConstConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   
flatten_74/ReshapeReshape$dropout_81/dropout/SelectV2:output:0flatten_74/Const:output:0*
T0*(
_output_shapes
:џџџџџџџџџ
%INJECTION_MASKS/MatMul/ReadVariableOpReadVariableOp.injection_masks_matmul_readvariableop_resource*
_output_shapes
:	*
dtype0
INJECTION_MASKS/MatMulMatMulflatten_74/Reshape:output:0-INJECTION_MASKS/MatMul/ReadVariableOp:value:0*
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
:џџџџџџџџџ
NoOpNoOp'^INJECTION_MASKS/BiasAdd/ReadVariableOp&^INJECTION_MASKS/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:џџџџџџџџџ:џџџџџџџџџ : : 2P
&INJECTION_MASKS/BiasAdd/ReadVariableOp&INJECTION_MASKS/BiasAdd/ReadVariableOp2N
%INJECTION_MASKS/MatMul/ReadVariableOp%INJECTION_MASKS/MatMul/ReadVariableOp:]Y
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
л

D__inference_model_74_layer_call_and_return_conditional_losses_282147
inputs_1

inputs)
injection_masks_282141:	$
injection_masks_282143:
identityЂ'INJECTION_MASKS/StatefulPartitionedCallЂ"dropout_81/StatefulPartitionedCallР
%whiten_passthrough_37/PartitionedCallPartitionedCallinputs_1*
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
)__inference_restored_function_body_282044л
reshape_74/PartitionedCallPartitionedCall.whiten_passthrough_37/PartitionedCall:output:0*
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
)__inference_restored_function_body_282050§
"dropout_81/StatefulPartitionedCallStatefulPartitionedCall#reshape_74/PartitionedCall:output:0*
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
  zE8 *O
fJRH
F__inference_dropout_81_layer_call_and_return_conditional_losses_282082ё
flatten_74/PartitionedCallPartitionedCall+dropout_81/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:џџџџџџџџџ* 
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
F__inference_flatten_74_layer_call_and_return_conditional_losses_282090И
'INJECTION_MASKS/StatefulPartitionedCallStatefulPartitionedCall#flatten_74/PartitionedCall:output:0injection_masks_282141injection_masks_282143*
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
K__inference_INJECTION_MASKS_layer_call_and_return_conditional_losses_282103
IdentityIdentity0INJECTION_MASKS/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ
NoOpNoOp(^INJECTION_MASKS/StatefulPartitionedCall#^dropout_81/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:џџџџџџџџџ:џџџџџџџџџ : : 2R
'INJECTION_MASKS/StatefulPartitionedCall'INJECTION_MASKS/StatefulPartitionedCall2H
"dropout_81/StatefulPartitionedCall"dropout_81/StatefulPartitionedCall:TP
,
_output_shapes
:џџџџџџџџџ 
 
_user_specified_nameinputs:U Q
-
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs"ѓ
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
StatefulPartitionedCall:0џџџџџџџџџtensorflow/serving/predict:лЇ
§
layer-0
layer-1
layer-2
layer-3
layer-4
layer-5
layer_with_weights-0
layer-6
	variables
	trainable_variables

regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
_default_save_signature
	optimizer

signatures
#_self_saveable_object_factories"
_tf_keras_network
D
#_self_saveable_object_factories"
_tf_keras_input_layer
Ъ
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
#_self_saveable_object_factories"
_tf_keras_layer
Ъ
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
# _self_saveable_object_factories"
_tf_keras_layer
с
!	variables
"trainable_variables
#regularization_losses
$	keras_api
%__call__
*&&call_and_return_all_conditional_losses
'_random_generator
#(_self_saveable_object_factories"
_tf_keras_layer
Ъ
)	variables
*trainable_variables
+regularization_losses
,	keras_api
-__call__
*.&call_and_return_all_conditional_losses
#/_self_saveable_object_factories"
_tf_keras_layer
D
#0_self_saveable_object_factories"
_tf_keras_input_layer
р
1	variables
2trainable_variables
3regularization_losses
4	keras_api
5__call__
*6&call_and_return_all_conditional_losses

7kernel
8bias
#9_self_saveable_object_factories"
_tf_keras_layer
.
70
81"
trackable_list_wrapper
.
70
81"
trackable_list_wrapper
 "
trackable_list_wrapper
Ъ
:non_trainable_variables

;layers
<metrics
=layer_regularization_losses
>layer_metrics
	variables
	trainable_variables

regularization_losses
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
Я
?trace_0
@trace_1
Atrace_2
Btrace_32ф
)__inference_model_74_layer_call_fn_282154
)__inference_model_74_layer_call_fn_282178
)__inference_model_74_layer_call_fn_282249
)__inference_model_74_layer_call_fn_282259Е
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
 z?trace_0z@trace_1zAtrace_2zBtrace_3
Л
Ctrace_0
Dtrace_1
Etrace_2
Ftrace_32а
D__inference_model_74_layer_call_and_return_conditional_losses_282110
D__inference_model_74_layer_call_and_return_conditional_losses_282129
D__inference_model_74_layer_call_and_return_conditional_losses_282283
D__inference_model_74_layer_call_and_return_conditional_losses_282300Е
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
 zCtrace_0zDtrace_1zEtrace_2zFtrace_3
иBе
!__inference__wrapped_model_282063	OFFSOURCEONSOURCE"
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
G
_variables
H_iterations
I_learning_rate
J_index_dict
K
_momentums
L_velocities
M_update_step_xla"
experimentalOptimizer
,
Nserving_default"
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
Onon_trainable_variables

Players
Qmetrics
Rlayer_regularization_losses
Slayer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
э
Ttrace_02а
3__inference_whiten_passthrough_37_layer_call_fn_381
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
 zTtrace_0

Utrace_02ы
N__inference_whiten_passthrough_37_layer_call_and_return_conditional_losses_484
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
 zUtrace_0
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
­
Vnon_trainable_variables

Wlayers
Xmetrics
Ylayer_regularization_losses
Zlayer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
т
[trace_02Х
(__inference_reshape_74_layer_call_fn_583
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
 z[trace_0
§
\trace_02р
C__inference_reshape_74_layer_call_and_return_conditional_losses_280
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
 z\trace_0
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
­
]non_trainable_variables

^layers
_metrics
`layer_regularization_losses
alayer_metrics
!	variables
"trainable_variables
#regularization_losses
%__call__
*&&call_and_return_all_conditional_losses
&&"call_and_return_conditional_losses"
_generic_user_object
Н
btrace_0
ctrace_12
+__inference_dropout_81_layer_call_fn_282315
+__inference_dropout_81_layer_call_fn_282320Љ
ЂВ
FullArgSpec!
args
jinputs

jtraining
varargs
 
varkw
 
defaultsЂ
p 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 zbtrace_0zctrace_1
ѓ
dtrace_0
etrace_12М
F__inference_dropout_81_layer_call_and_return_conditional_losses_282332
F__inference_dropout_81_layer_call_and_return_conditional_losses_282337Љ
ЂВ
FullArgSpec!
args
jinputs

jtraining
varargs
 
varkw
 
defaultsЂ
p 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 zdtrace_0zetrace_1
C
#f_self_saveable_object_factories"
_generic_user_object
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
­
gnon_trainable_variables

hlayers
imetrics
jlayer_regularization_losses
klayer_metrics
)	variables
*trainable_variables
+regularization_losses
-__call__
*.&call_and_return_all_conditional_losses
&."call_and_return_conditional_losses"
_generic_user_object
х
ltrace_02Ш
+__inference_flatten_74_layer_call_fn_282342
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
 zltrace_0

mtrace_02у
F__inference_flatten_74_layer_call_and_return_conditional_losses_282348
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
 zmtrace_0
 "
trackable_dict_wrapper
 "
trackable_dict_wrapper
.
70
81"
trackable_list_wrapper
.
70
81"
trackable_list_wrapper
 "
trackable_list_wrapper
­
nnon_trainable_variables

olayers
pmetrics
qlayer_regularization_losses
rlayer_metrics
1	variables
2trainable_variables
3regularization_losses
5__call__
*6&call_and_return_all_conditional_losses
&6"call_and_return_conditional_losses"
_generic_user_object
ъ
strace_02Э
0__inference_INJECTION_MASKS_layer_call_fn_282357
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
 zstrace_0

ttrace_02ш
K__inference_INJECTION_MASKS_layer_call_and_return_conditional_losses_282368
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
 zttrace_0
:	 2kernel
: 2bias
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
Q
0
1
2
3
4
5
6"
trackable_list_wrapper
.
u0
v1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
§Bњ
)__inference_model_74_layer_call_fn_282154	OFFSOURCEONSOURCE"Е
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
)__inference_model_74_layer_call_fn_282178	OFFSOURCEONSOURCE"Е
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
)__inference_model_74_layer_call_fn_282249inputs_offsourceinputs_onsource"Е
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
)__inference_model_74_layer_call_fn_282259inputs_offsourceinputs_onsource"Е
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
D__inference_model_74_layer_call_and_return_conditional_losses_282110	OFFSOURCEONSOURCE"Е
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
D__inference_model_74_layer_call_and_return_conditional_losses_282129	OFFSOURCEONSOURCE"Е
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
D__inference_model_74_layer_call_and_return_conditional_losses_282283inputs_offsourceinputs_onsource"Е
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
D__inference_model_74_layer_call_and_return_conditional_losses_282300inputs_offsourceinputs_onsource"Е
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
C
H0
w1
x2
y3
z4"
trackable_list_wrapper
:	  2	iteration
:  2learning_rate
 "
trackable_dict_wrapper
.
w0
y1"
trackable_list_wrapper
.
x0
z1"
trackable_list_wrapper
Г
{trace_0
|trace_12ќ
#__inference__update_step_xla_282305
#__inference__update_step_xla_282310Џ
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
 0z{trace_0z|trace_1
еBв
$__inference_signature_wrapper_282239	OFFSOURCEONSOURCE"
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
3__inference_whiten_passthrough_37_layer_call_fn_381inputs"
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
N__inference_whiten_passthrough_37_layer_call_and_return_conditional_losses_484inputs"
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
(__inference_reshape_74_layer_call_fn_583inputs"
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
C__inference_reshape_74_layer_call_and_return_conditional_losses_280inputs"
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
цBу
+__inference_dropout_81_layer_call_fn_282315inputs"Љ
ЂВ
FullArgSpec!
args
jinputs

jtraining
varargs
 
varkw
 
defaultsЂ
p 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
цBу
+__inference_dropout_81_layer_call_fn_282320inputs"Љ
ЂВ
FullArgSpec!
args
jinputs

jtraining
varargs
 
varkw
 
defaultsЂ
p 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
Bў
F__inference_dropout_81_layer_call_and_return_conditional_losses_282332inputs"Љ
ЂВ
FullArgSpec!
args
jinputs

jtraining
varargs
 
varkw
 
defaultsЂ
p 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
Bў
F__inference_dropout_81_layer_call_and_return_conditional_losses_282337inputs"Љ
ЂВ
FullArgSpec!
args
jinputs

jtraining
varargs
 
varkw
 
defaultsЂ
p 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
 "
trackable_dict_wrapper
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
+__inference_flatten_74_layer_call_fn_282342inputs"
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
F__inference_flatten_74_layer_call_and_return_conditional_losses_282348inputs"
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
0__inference_INJECTION_MASKS_layer_call_fn_282357inputs"
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
K__inference_INJECTION_MASKS_layer_call_and_return_conditional_losses_282368inputs"
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
O
}	variables
~	keras_api
	total

count"
_tf_keras_metric
c
	variables
	keras_api

total

count

_fn_kwargs"
_tf_keras_metric
 :	 2Adam/m/kernel
 :	 2Adam/v/kernel
: 2Adam/m/bias
: 2Adam/v/bias
юBы
#__inference__update_step_xla_282305gradientvariable"­
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
#__inference__update_step_xla_282310gradientvariable"­
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
/
0
1"
trackable_list_wrapper
-
}	variables"
_generic_user_object
:  (2total
:  (2count
0
0
1"
trackable_list_wrapper
.
	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapperГ
K__inference_INJECTION_MASKS_layer_call_and_return_conditional_losses_282368d780Ђ-
&Ђ#
!
inputsџџџџџџџџџ
Њ ",Ђ)
"
tensor_0џџџџџџџџџ
 
0__inference_INJECTION_MASKS_layer_call_fn_282357Y780Ђ-
&Ђ#
!
inputsџџџџџџџџџ
Њ "!
unknownџџџџџџџџџ
#__inference__update_step_xla_282305pjЂg
`Ђ]

gradient	
52	Ђ
њ	

p
` VariableSpec 
`рЪо?
Њ "
 
#__inference__update_step_xla_282310f`Ђ]
VЂS

gradient
0-	Ђ
њ

p
` VariableSpec 
`ртЧо?
Њ "
 ю
!__inference__wrapped_model_282063Ш78Ђ|
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
injection_masksџџџџџџџџџЗ
F__inference_dropout_81_layer_call_and_return_conditional_losses_282332m8Ђ5
.Ђ+
%"
inputsџџџџџџџџџ
p
Њ "1Ђ.
'$
tensor_0џџџџџџџџџ
 З
F__inference_dropout_81_layer_call_and_return_conditional_losses_282337m8Ђ5
.Ђ+
%"
inputsџџџџџџџџџ
p 
Њ "1Ђ.
'$
tensor_0џџџџџџџџџ
 
+__inference_dropout_81_layer_call_fn_282315b8Ђ5
.Ђ+
%"
inputsџџџџџџџџџ
p
Њ "&#
unknownџџџџџџџџџ
+__inference_dropout_81_layer_call_fn_282320b8Ђ5
.Ђ+
%"
inputsџџџџџџџџџ
p 
Њ "&#
unknownџџџџџџџџџЏ
F__inference_flatten_74_layer_call_and_return_conditional_losses_282348e4Ђ1
*Ђ'
%"
inputsџџџџџџџџџ
Њ "-Ђ*
# 
tensor_0џџџџџџџџџ
 
+__inference_flatten_74_layer_call_fn_282342Z4Ђ1
*Ђ'
%"
inputsџџџџџџџџџ
Њ ""
unknownџџџџџџџџџ
D__inference_model_74_layer_call_and_return_conditional_losses_282110Н78Ђ
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
 
D__inference_model_74_layer_call_and_return_conditional_losses_282129Н78Ђ
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
 
D__inference_model_74_layer_call_and_return_conditional_losses_282283Э78Ђ
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
 
D__inference_model_74_layer_call_and_return_conditional_losses_282300Э78Ђ
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
 р
)__inference_model_74_layer_call_fn_282154В78Ђ
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
unknownџџџџџџџџџр
)__inference_model_74_layer_call_fn_282178В78Ђ
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
unknownџџџџџџџџџ№
)__inference_model_74_layer_call_fn_282249Т78Ђ
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
unknownџџџџџџџџџ№
)__inference_model_74_layer_call_fn_282259Т78Ђ
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
C__inference_reshape_74_layer_call_and_return_conditional_losses_280i4Ђ1
*Ђ'
%"
inputsџџџџџџџџџ
Њ "1Ђ.
'$
tensor_0џџџџџџџџџ
 
(__inference_reshape_74_layer_call_fn_583^4Ђ1
*Ђ'
%"
inputsџџџџџџџџџ
Њ "&#
unknownџџџџџџџџџь
$__inference_signature_wrapper_282239У78zЂw
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
N__inference_whiten_passthrough_37_layer_call_and_return_conditional_losses_484j5Ђ2
+Ђ(
&#
inputsџџџџџџџџџ
Њ "1Ђ.
'$
tensor_0џџџџџџџџџ
 
3__inference_whiten_passthrough_37_layer_call_fn_381_5Ђ2
+Ђ(
&#
inputsџџџџџџџџџ
Њ "&#
unknownџџџџџџџџџ