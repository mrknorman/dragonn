Ѕ
О
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
h
ConcatV2
values"T*N
axis"Tidx
output"T"
Nint(0"	
Ttype"
Tidxtype0:
2	
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
Ў
GatherV2
params"Tparams
indices"Tindices
axis"Taxis
output"Tparams"

batch_dimsint "
Tparamstype"
Tindicestype:
2	"
Taxistype:
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

Prod

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( ""
Ttype:
2	"
Tidxtype0:
2	
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
 "serve*2.12.12v2.12.0-25-g8e2b6655c0c8рЭ
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
:k*
shared_nameAdam/v/kernel
o
!Adam/v/kernel/Read/ReadVariableOpReadVariableOpAdam/v/kernel*
_output_shapes

:k*
dtype0
v
Adam/m/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:k*
shared_nameAdam/m/kernel
o
!Adam/m/kernel/Read/ReadVariableOpReadVariableOpAdam/m/kernel*
_output_shapes

:k*
dtype0
r
Adam/v/bias_1VarHandleOp*
_output_shapes
: *
dtype0*
shape:k*
shared_nameAdam/v/bias_1
k
!Adam/v/bias_1/Read/ReadVariableOpReadVariableOpAdam/v/bias_1*
_output_shapes
:k*
dtype0
r
Adam/m/bias_1VarHandleOp*
_output_shapes
: *
dtype0*
shape:k*
shared_nameAdam/m/bias_1
k
!Adam/m/bias_1/Read/ReadVariableOpReadVariableOpAdam/m/bias_1*
_output_shapes
:k*
dtype0
~
Adam/v/kernel_1VarHandleOp*
_output_shapes
: *
dtype0*
shape:sak* 
shared_nameAdam/v/kernel_1
w
#Adam/v/kernel_1/Read/ReadVariableOpReadVariableOpAdam/v/kernel_1*"
_output_shapes
:sak*
dtype0
~
Adam/m/kernel_1VarHandleOp*
_output_shapes
: *
dtype0*
shape:sak* 
shared_nameAdam/m/kernel_1
w
#Adam/m/kernel_1/Read/ReadVariableOpReadVariableOpAdam/m/kernel_1*"
_output_shapes
:sak*
dtype0
r
Adam/v/bias_2VarHandleOp*
_output_shapes
: *
dtype0*
shape:a*
shared_nameAdam/v/bias_2
k
!Adam/v/bias_2/Read/ReadVariableOpReadVariableOpAdam/v/bias_2*
_output_shapes
:a*
dtype0
r
Adam/m/bias_2VarHandleOp*
_output_shapes
: *
dtype0*
shape:a*
shared_nameAdam/m/bias_2
k
!Adam/m/bias_2/Read/ReadVariableOpReadVariableOpAdam/m/bias_2*
_output_shapes
:a*
dtype0
~
Adam/v/kernel_2VarHandleOp*
_output_shapes
: *
dtype0*
shape:ja* 
shared_nameAdam/v/kernel_2
w
#Adam/v/kernel_2/Read/ReadVariableOpReadVariableOpAdam/v/kernel_2*"
_output_shapes
:ja*
dtype0
~
Adam/m/kernel_2VarHandleOp*
_output_shapes
: *
dtype0*
shape:ja* 
shared_nameAdam/m/kernel_2
w
#Adam/m/kernel_2/Read/ReadVariableOpReadVariableOpAdam/m/kernel_2*"
_output_shapes
:ja*
dtype0
r
Adam/v/bias_3VarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameAdam/v/bias_3
k
!Adam/v/bias_3/Read/ReadVariableOpReadVariableOpAdam/v/bias_3*
_output_shapes
:*
dtype0
r
Adam/m/bias_3VarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameAdam/m/bias_3
k
!Adam/m/bias_3/Read/ReadVariableOpReadVariableOpAdam/m/bias_3*
_output_shapes
:*
dtype0
z
Adam/v/kernel_3VarHandleOp*
_output_shapes
: *
dtype0*
shape
:y* 
shared_nameAdam/v/kernel_3
s
#Adam/v/kernel_3/Read/ReadVariableOpReadVariableOpAdam/v/kernel_3*
_output_shapes

:y*
dtype0
z
Adam/m/kernel_3VarHandleOp*
_output_shapes
: *
dtype0*
shape
:y* 
shared_nameAdam/m/kernel_3
s
#Adam/m/kernel_3/Read/ReadVariableOpReadVariableOpAdam/m/kernel_3*
_output_shapes

:y*
dtype0
r
Adam/v/bias_4VarHandleOp*
_output_shapes
: *
dtype0*
shape:y*
shared_nameAdam/v/bias_4
k
!Adam/v/bias_4/Read/ReadVariableOpReadVariableOpAdam/v/bias_4*
_output_shapes
:y*
dtype0
r
Adam/m/bias_4VarHandleOp*
_output_shapes
: *
dtype0*
shape:y*
shared_nameAdam/m/bias_4
k
!Adam/m/bias_4/Read/ReadVariableOpReadVariableOpAdam/m/bias_4*
_output_shapes
:y*
dtype0
~
Adam/v/kernel_4VarHandleOp*
_output_shapes
: *
dtype0*
shape:Fy* 
shared_nameAdam/v/kernel_4
w
#Adam/v/kernel_4/Read/ReadVariableOpReadVariableOpAdam/v/kernel_4*"
_output_shapes
:Fy*
dtype0
~
Adam/m/kernel_4VarHandleOp*
_output_shapes
: *
dtype0*
shape:Fy* 
shared_nameAdam/m/kernel_4
w
#Adam/m/kernel_4/Read/ReadVariableOpReadVariableOpAdam/m/kernel_4*"
_output_shapes
:Fy*
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
:k*
shared_namekernel
a
kernel/Read/ReadVariableOpReadVariableOpkernel*
_output_shapes

:k*
dtype0
d
bias_1VarHandleOp*
_output_shapes
: *
dtype0*
shape:k*
shared_namebias_1
]
bias_1/Read/ReadVariableOpReadVariableOpbias_1*
_output_shapes
:k*
dtype0
p
kernel_1VarHandleOp*
_output_shapes
: *
dtype0*
shape:sak*
shared_name
kernel_1
i
kernel_1/Read/ReadVariableOpReadVariableOpkernel_1*"
_output_shapes
:sak*
dtype0
d
bias_2VarHandleOp*
_output_shapes
: *
dtype0*
shape:a*
shared_namebias_2
]
bias_2/Read/ReadVariableOpReadVariableOpbias_2*
_output_shapes
:a*
dtype0
p
kernel_2VarHandleOp*
_output_shapes
: *
dtype0*
shape:ja*
shared_name
kernel_2
i
kernel_2/Read/ReadVariableOpReadVariableOpkernel_2*"
_output_shapes
:ja*
dtype0
d
bias_3VarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namebias_3
]
bias_3/Read/ReadVariableOpReadVariableOpbias_3*
_output_shapes
:*
dtype0
l
kernel_3VarHandleOp*
_output_shapes
: *
dtype0*
shape
:y*
shared_name
kernel_3
e
kernel_3/Read/ReadVariableOpReadVariableOpkernel_3*
_output_shapes

:y*
dtype0
d
bias_4VarHandleOp*
_output_shapes
: *
dtype0*
shape:y*
shared_namebias_4
]
bias_4/Read/ReadVariableOpReadVariableOpbias_4*
_output_shapes
:y*
dtype0
p
kernel_4VarHandleOp*
_output_shapes
: *
dtype0*
shape:Fy*
shared_name
kernel_4
i
kernel_4/Read/ReadVariableOpReadVariableOpkernel_4*"
_output_shapes
:Fy*
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
Ш
StatefulPartitionedCallStatefulPartitionedCallserving_default_OFFSOURCEserving_default_ONSOURCEkernel_4bias_4kernel_3bias_3kernel_2bias_2kernel_1bias_1kernelbias*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*,
_read_only_resource_inputs

	
*<
config_proto,*

CPU

GPU(2*0J

  zE8 *-
f(R&
$__inference_signature_wrapper_799461

NoOpNoOp
фU
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*U
valueUBU BU

layer-0
layer-1
layer-2
layer_with_weights-0
layer-3
layer-4
layer_with_weights-1
layer-5
layer_with_weights-2
layer-6
layer_with_weights-3
layer-7
	layer-8

layer-9
layer_with_weights-4
layer-10
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
_default_save_signature
	optimizer

signatures
#_self_saveable_object_factories*
'
#_self_saveable_object_factories* 
Г
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
#_self_saveable_object_factories* 
Г
	variables
trainable_variables
 regularization_losses
!	keras_api
"__call__
*#&call_and_return_all_conditional_losses
#$_self_saveable_object_factories* 
э
%	variables
&trainable_variables
'regularization_losses
(	keras_api
)__call__
**&call_and_return_all_conditional_losses

+kernel
,bias
#-_self_saveable_object_factories
 ._jit_compiled_convolution_op*
Ъ
/	variables
0trainable_variables
1regularization_losses
2	keras_api
3__call__
*4&call_and_return_all_conditional_losses
5_random_generator
#6_self_saveable_object_factories* 
Ы
7	variables
8trainable_variables
9regularization_losses
:	keras_api
;__call__
*<&call_and_return_all_conditional_losses

=kernel
>bias
#?_self_saveable_object_factories*
э
@	variables
Atrainable_variables
Bregularization_losses
C	keras_api
D__call__
*E&call_and_return_all_conditional_losses

Fkernel
Gbias
#H_self_saveable_object_factories
 I_jit_compiled_convolution_op*
э
J	variables
Ktrainable_variables
Lregularization_losses
M	keras_api
N__call__
*O&call_and_return_all_conditional_losses

Pkernel
Qbias
#R_self_saveable_object_factories
 S_jit_compiled_convolution_op*
Г
T	variables
Utrainable_variables
Vregularization_losses
W	keras_api
X__call__
*Y&call_and_return_all_conditional_losses
#Z_self_saveable_object_factories* 
'
#[_self_saveable_object_factories* 
Ы
\	variables
]trainable_variables
^regularization_losses
_	keras_api
`__call__
*a&call_and_return_all_conditional_losses

bkernel
cbias
#d_self_saveable_object_factories*
J
+0
,1
=2
>3
F4
G5
P6
Q7
b8
c9*
J
+0
,1
=2
>3
F4
G5
P6
Q7
b8
c9*
* 
А
enon_trainable_variables

flayers
gmetrics
hlayer_regularization_losses
ilayer_metrics
	variables
trainable_variables
regularization_losses
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*
6
jtrace_0
ktrace_1
ltrace_2
mtrace_3* 
6
ntrace_0
otrace_1
ptrace_2
qtrace_3* 
* 

r
_variables
s_iterations
t_learning_rate
u_index_dict
v
_momentums
w_velocities
x_update_step_xla*

yserving_default* 
* 
* 
* 
* 
* 

znon_trainable_variables

{layers
|metrics
}layer_regularization_losses
~layer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses* 

trace_0* 

trace_0* 
* 
* 
* 
* 

non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
	variables
trainable_variables
 regularization_losses
"__call__
*#&call_and_return_all_conditional_losses
&#"call_and_return_conditional_losses* 

trace_0* 

trace_0* 
* 

+0
,1*

+0
,1*
* 

non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
%	variables
&trainable_variables
'regularization_losses
)__call__
**&call_and_return_all_conditional_losses
&*"call_and_return_conditional_losses*

trace_0* 

trace_0* 
XR
VARIABLE_VALUEkernel_46layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE*
TN
VARIABLE_VALUEbias_44layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
* 
* 

non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
/	variables
0trainable_variables
1regularization_losses
3__call__
*4&call_and_return_all_conditional_losses
&4"call_and_return_conditional_losses* 

trace_0
trace_1* 

trace_0
trace_1* 
(
$_self_saveable_object_factories* 
* 

=0
>1*

=0
>1*
* 

non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
7	variables
8trainable_variables
9regularization_losses
;__call__
*<&call_and_return_all_conditional_losses
&<"call_and_return_conditional_losses*

trace_0* 

trace_0* 
XR
VARIABLE_VALUEkernel_36layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE*
TN
VARIABLE_VALUEbias_34layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 

F0
G1*

F0
G1*
* 

 non_trainable_variables
Ёlayers
Ђmetrics
 Ѓlayer_regularization_losses
Єlayer_metrics
@	variables
Atrainable_variables
Bregularization_losses
D__call__
*E&call_and_return_all_conditional_losses
&E"call_and_return_conditional_losses*

Ѕtrace_0* 

Іtrace_0* 
XR
VARIABLE_VALUEkernel_26layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE*
TN
VARIABLE_VALUEbias_24layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 

P0
Q1*

P0
Q1*
* 

Їnon_trainable_variables
Јlayers
Љmetrics
 Њlayer_regularization_losses
Ћlayer_metrics
J	variables
Ktrainable_variables
Lregularization_losses
N__call__
*O&call_and_return_all_conditional_losses
&O"call_and_return_conditional_losses*

Ќtrace_0* 

­trace_0* 
XR
VARIABLE_VALUEkernel_16layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUE*
TN
VARIABLE_VALUEbias_14layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
* 
* 

Ўnon_trainable_variables
Џlayers
Аmetrics
 Бlayer_regularization_losses
Вlayer_metrics
T	variables
Utrainable_variables
Vregularization_losses
X__call__
*Y&call_and_return_all_conditional_losses
&Y"call_and_return_conditional_losses* 

Гtrace_0* 

Дtrace_0* 
* 
* 

b0
c1*

b0
c1*
* 

Еnon_trainable_variables
Жlayers
Зmetrics
 Иlayer_regularization_losses
Йlayer_metrics
\	variables
]trainable_variables
^regularization_losses
`__call__
*a&call_and_return_all_conditional_losses
&a"call_and_return_conditional_losses*

Кtrace_0* 

Лtrace_0* 
VP
VARIABLE_VALUEkernel6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUE*
RL
VARIABLE_VALUEbias4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
R
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
9
10*

М0
Н1*
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
Ж
s0
О1
П2
Р3
С4
Т5
У6
Ф7
Х8
Ц9
Ч10
Ш11
Щ12
Ъ13
Ы14
Ь15
Э16
Ю17
Я18
а19
б20*
SM
VARIABLE_VALUE	iteration0optimizer/_iterations/.ATTRIBUTES/VARIABLE_VALUE*
ZT
VARIABLE_VALUElearning_rate3optimizer/_learning_rate/.ATTRIBUTES/VARIABLE_VALUE*
* 
T
О0
Р1
Т2
Ф3
Ц4
Ш5
Ъ6
Ь7
Ю8
а9*
T
П0
С1
У2
Х3
Ч4
Щ5
Ы6
Э7
Я8
б9*

вtrace_0
гtrace_1
дtrace_2
еtrace_3
жtrace_4
зtrace_5
иtrace_6
йtrace_7
кtrace_8
лtrace_9* 
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
м	variables
н	keras_api

оtotal

пcount*
M
р	variables
с	keras_api

тtotal

уcount
ф
_fn_kwargs*
ZT
VARIABLE_VALUEAdam/m/kernel_41optimizer/_variables/1/.ATTRIBUTES/VARIABLE_VALUE*
ZT
VARIABLE_VALUEAdam/v/kernel_41optimizer/_variables/2/.ATTRIBUTES/VARIABLE_VALUE*
XR
VARIABLE_VALUEAdam/m/bias_41optimizer/_variables/3/.ATTRIBUTES/VARIABLE_VALUE*
XR
VARIABLE_VALUEAdam/v/bias_41optimizer/_variables/4/.ATTRIBUTES/VARIABLE_VALUE*
ZT
VARIABLE_VALUEAdam/m/kernel_31optimizer/_variables/5/.ATTRIBUTES/VARIABLE_VALUE*
ZT
VARIABLE_VALUEAdam/v/kernel_31optimizer/_variables/6/.ATTRIBUTES/VARIABLE_VALUE*
XR
VARIABLE_VALUEAdam/m/bias_31optimizer/_variables/7/.ATTRIBUTES/VARIABLE_VALUE*
XR
VARIABLE_VALUEAdam/v/bias_31optimizer/_variables/8/.ATTRIBUTES/VARIABLE_VALUE*
ZT
VARIABLE_VALUEAdam/m/kernel_21optimizer/_variables/9/.ATTRIBUTES/VARIABLE_VALUE*
[U
VARIABLE_VALUEAdam/v/kernel_22optimizer/_variables/10/.ATTRIBUTES/VARIABLE_VALUE*
YS
VARIABLE_VALUEAdam/m/bias_22optimizer/_variables/11/.ATTRIBUTES/VARIABLE_VALUE*
YS
VARIABLE_VALUEAdam/v/bias_22optimizer/_variables/12/.ATTRIBUTES/VARIABLE_VALUE*
[U
VARIABLE_VALUEAdam/m/kernel_12optimizer/_variables/13/.ATTRIBUTES/VARIABLE_VALUE*
[U
VARIABLE_VALUEAdam/v/kernel_12optimizer/_variables/14/.ATTRIBUTES/VARIABLE_VALUE*
YS
VARIABLE_VALUEAdam/m/bias_12optimizer/_variables/15/.ATTRIBUTES/VARIABLE_VALUE*
YS
VARIABLE_VALUEAdam/v/bias_12optimizer/_variables/16/.ATTRIBUTES/VARIABLE_VALUE*
YS
VARIABLE_VALUEAdam/m/kernel2optimizer/_variables/17/.ATTRIBUTES/VARIABLE_VALUE*
YS
VARIABLE_VALUEAdam/v/kernel2optimizer/_variables/18/.ATTRIBUTES/VARIABLE_VALUE*
WQ
VARIABLE_VALUEAdam/m/bias2optimizer/_variables/19/.ATTRIBUTES/VARIABLE_VALUE*
WQ
VARIABLE_VALUEAdam/v/bias2optimizer/_variables/20/.ATTRIBUTES/VARIABLE_VALUE*
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

о0
п1*

м	variables*
UO
VARIABLE_VALUEtotal_14keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE*
UO
VARIABLE_VALUEcount_14keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE*

т0
у1*

р	variables*
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

StatefulPartitionedCall_1StatefulPartitionedCallsaver_filenamekernel_4bias_4kernel_3bias_3kernel_2bias_2kernel_1bias_1kernelbias	iterationlearning_rateAdam/m/kernel_4Adam/v/kernel_4Adam/m/bias_4Adam/v/bias_4Adam/m/kernel_3Adam/v/kernel_3Adam/m/bias_3Adam/v/bias_3Adam/m/kernel_2Adam/v/kernel_2Adam/m/bias_2Adam/v/bias_2Adam/m/kernel_1Adam/v/kernel_1Adam/m/bias_1Adam/v/bias_1Adam/m/kernelAdam/v/kernelAdam/m/biasAdam/v/biastotal_1count_1totalcountConst*1
Tin*
(2&*
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
__inference__traced_save_800421

StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamekernel_4bias_4kernel_3bias_3kernel_2bias_2kernel_1bias_1kernelbias	iterationlearning_rateAdam/m/kernel_4Adam/v/kernel_4Adam/m/bias_4Adam/v/bias_4Adam/m/kernel_3Adam/v/kernel_3Adam/m/bias_3Adam/v/bias_3Adam/m/kernel_2Adam/v/kernel_2Adam/m/bias_2Adam/v/bias_2Adam/m/kernel_1Adam/v/kernel_1Adam/m/bias_1Adam/v/bias_1Adam/m/kernelAdam/v/kernelAdam/m/biasAdam/v/biastotal_1count_1totalcount*0
Tin)
'2%*
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
"__inference__traced_restore_800539 Ї
с

)__inference_dense_89_layer_call_fn_799773

inputs
unknown:y
	unknown_0:
identityЂStatefulPartitionedCallь
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:џџџџџџџџџ2*$
_read_only_resource_inputs
*<
config_proto,*

CPU

GPU(2*0J

  zE8 *M
fHRF
D__inference_dense_89_layer_call_and_return_conditional_losses_799060s
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:џџџџџџџџџ2`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:џџџџџџџџџ2y: : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:џџџџџџџџџ2y
 
_user_specified_nameinputs
Јv
	
D__inference_model_76_layer_call_and_return_conditional_losses_799616
inputs_offsource
inputs_onsourceK
5conv1d_81_conv1d_expanddims_1_readvariableop_resource:Fy7
)conv1d_81_biasadd_readvariableop_resource:y<
*dense_89_tensordot_readvariableop_resource:y6
(dense_89_biasadd_readvariableop_resource:K
5conv1d_82_conv1d_expanddims_1_readvariableop_resource:ja7
)conv1d_82_biasadd_readvariableop_resource:aK
5conv1d_83_conv1d_expanddims_1_readvariableop_resource:sak7
)conv1d_83_biasadd_readvariableop_resource:k@
.injection_masks_matmul_readvariableop_resource:k=
/injection_masks_biasadd_readvariableop_resource:
identityЂ&INJECTION_MASKS/BiasAdd/ReadVariableOpЂ%INJECTION_MASKS/MatMul/ReadVariableOpЂ conv1d_81/BiasAdd/ReadVariableOpЂ,conv1d_81/Conv1D/ExpandDims_1/ReadVariableOpЂ conv1d_82/BiasAdd/ReadVariableOpЂ,conv1d_82/Conv1D/ExpandDims_1/ReadVariableOpЂ conv1d_83/BiasAdd/ReadVariableOpЂ,conv1d_83/Conv1D/ExpandDims_1/ReadVariableOpЂdense_89/BiasAdd/ReadVariableOpЂ!dense_89/Tensordot/ReadVariableOpШ
%whiten_passthrough_38/PartitionedCallPartitionedCallinputs_offsource*
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
)__inference_restored_function_body_285667л
reshape_76/PartitionedCallPartitionedCall.whiten_passthrough_38/PartitionedCall:output:0*
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
)__inference_restored_function_body_285673j
conv1d_81/Conv1D/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
§џџџџџџџџГ
conv1d_81/Conv1D/ExpandDims
ExpandDims#reshape_76/PartitionedCall:output:0(conv1d_81/Conv1D/ExpandDims/dim:output:0*
T0*0
_output_shapes
:џџџџџџџџџІ
,conv1d_81/Conv1D/ExpandDims_1/ReadVariableOpReadVariableOp5conv1d_81_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:Fy*
dtype0c
!conv1d_81/Conv1D/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : О
conv1d_81/Conv1D/ExpandDims_1
ExpandDims4conv1d_81/Conv1D/ExpandDims_1/ReadVariableOp:value:0*conv1d_81/Conv1D/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:FyЪ
conv1d_81/Conv1DConv2D$conv1d_81/Conv1D/ExpandDims:output:0&conv1d_81/Conv1D/ExpandDims_1:output:0*
T0*/
_output_shapes
:џџџџџџџџџ2y*
paddingSAME*
strides
)
conv1d_81/Conv1D/SqueezeSqueezeconv1d_81/Conv1D:output:0*
T0*+
_output_shapes
:џџџџџџџџџ2y*
squeeze_dims

§џџџџџџџџ
 conv1d_81/BiasAdd/ReadVariableOpReadVariableOp)conv1d_81_biasadd_readvariableop_resource*
_output_shapes
:y*
dtype0
conv1d_81/BiasAddBiasAdd!conv1d_81/Conv1D/Squeeze:output:0(conv1d_81/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:џџџџџџџџџ2yn
conv1d_81/SigmoidSigmoidconv1d_81/BiasAdd:output:0*
T0*+
_output_shapes
:џџџџџџџџџ2y]
dropout_83/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *6Щ?
dropout_83/dropout/MulMulconv1d_81/Sigmoid:y:0!dropout_83/dropout/Const:output:0*
T0*+
_output_shapes
:џџџџџџџџџ2yk
dropout_83/dropout/ShapeShapeconv1d_81/Sigmoid:y:0*
T0*
_output_shapes
::эЯГ
/dropout_83/dropout/random_uniform/RandomUniformRandomUniform!dropout_83/dropout/Shape:output:0*
T0*+
_output_shapes
:џџџџџџџџџ2y*
dtype0*
seedшf
!dropout_83/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 * KК>Ы
dropout_83/dropout/GreaterEqualGreaterEqual8dropout_83/dropout/random_uniform/RandomUniform:output:0*dropout_83/dropout/GreaterEqual/y:output:0*
T0*+
_output_shapes
:џџџџџџџџџ2y_
dropout_83/dropout/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    У
dropout_83/dropout/SelectV2SelectV2#dropout_83/dropout/GreaterEqual:z:0dropout_83/dropout/Mul:z:0#dropout_83/dropout/Const_1:output:0*
T0*+
_output_shapes
:џџџџџџџџџ2y
!dense_89/Tensordot/ReadVariableOpReadVariableOp*dense_89_tensordot_readvariableop_resource*
_output_shapes

:y*
dtype0a
dense_89/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:h
dense_89/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       z
dense_89/Tensordot/ShapeShape$dropout_83/dropout/SelectV2:output:0*
T0*
_output_shapes
::эЯb
 dense_89/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : п
dense_89/Tensordot/GatherV2GatherV2!dense_89/Tensordot/Shape:output:0 dense_89/Tensordot/free:output:0)dense_89/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:d
"dense_89/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : у
dense_89/Tensordot/GatherV2_1GatherV2!dense_89/Tensordot/Shape:output:0 dense_89/Tensordot/axes:output:0+dense_89/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:b
dense_89/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 
dense_89/Tensordot/ProdProd$dense_89/Tensordot/GatherV2:output:0!dense_89/Tensordot/Const:output:0*
T0*
_output_shapes
: d
dense_89/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 
dense_89/Tensordot/Prod_1Prod&dense_89/Tensordot/GatherV2_1:output:0#dense_89/Tensordot/Const_1:output:0*
T0*
_output_shapes
: `
dense_89/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : Р
dense_89/Tensordot/concatConcatV2 dense_89/Tensordot/free:output:0 dense_89/Tensordot/axes:output:0'dense_89/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:
dense_89/Tensordot/stackPack dense_89/Tensordot/Prod:output:0"dense_89/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:Љ
dense_89/Tensordot/transpose	Transpose$dropout_83/dropout/SelectV2:output:0"dense_89/Tensordot/concat:output:0*
T0*+
_output_shapes
:џџџџџџџџџ2yЅ
dense_89/Tensordot/ReshapeReshape dense_89/Tensordot/transpose:y:0!dense_89/Tensordot/stack:output:0*
T0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџЅ
dense_89/Tensordot/MatMulMatMul#dense_89/Tensordot/Reshape:output:0)dense_89/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџd
dense_89/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:b
 dense_89/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : Ы
dense_89/Tensordot/concat_1ConcatV2$dense_89/Tensordot/GatherV2:output:0#dense_89/Tensordot/Const_2:output:0)dense_89/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:
dense_89/TensordotReshape#dense_89/Tensordot/MatMul:product:0$dense_89/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:џџџџџџџџџ2
dense_89/BiasAdd/ReadVariableOpReadVariableOp(dense_89_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
dense_89/BiasAddBiasAdddense_89/Tensordot:output:0'dense_89/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:џџџџџџџџџ2l
dense_89/SigmoidSigmoiddense_89/BiasAdd:output:0*
T0*+
_output_shapes
:џџџџџџџџџ2j
conv1d_82/Conv1D/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
§џџџџџџџџЃ
conv1d_82/Conv1D/ExpandDims
ExpandDimsdense_89/Sigmoid:y:0(conv1d_82/Conv1D/ExpandDims/dim:output:0*
T0*/
_output_shapes
:џџџџџџџџџ2І
,conv1d_82/Conv1D/ExpandDims_1/ReadVariableOpReadVariableOp5conv1d_82_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:ja*
dtype0c
!conv1d_82/Conv1D/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : О
conv1d_82/Conv1D/ExpandDims_1
ExpandDims4conv1d_82/Conv1D/ExpandDims_1/ReadVariableOp:value:0*conv1d_82/Conv1D/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:jaЪ
conv1d_82/Conv1DConv2D$conv1d_82/Conv1D/ExpandDims:output:0&conv1d_82/Conv1D/ExpandDims_1:output:0*
T0*/
_output_shapes
:џџџџџџџџџa*
paddingSAME*
strides
=
conv1d_82/Conv1D/SqueezeSqueezeconv1d_82/Conv1D:output:0*
T0*+
_output_shapes
:џџџџџџџџџa*
squeeze_dims

§џџџџџџџџ
 conv1d_82/BiasAdd/ReadVariableOpReadVariableOp)conv1d_82_biasadd_readvariableop_resource*
_output_shapes
:a*
dtype0
conv1d_82/BiasAddBiasAdd!conv1d_82/Conv1D/Squeeze:output:0(conv1d_82/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:џџџџџџџџџaS
conv1d_82/betaConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
conv1d_82/mulMulconv1d_82/beta:output:0conv1d_82/BiasAdd:output:0*
T0*+
_output_shapes
:џџџџџџџџџae
conv1d_82/SigmoidSigmoidconv1d_82/mul:z:0*
T0*+
_output_shapes
:џџџџџџџџџa
conv1d_82/mul_1Mulconv1d_82/BiasAdd:output:0conv1d_82/Sigmoid:y:0*
T0*+
_output_shapes
:џџџџџџџџџai
conv1d_82/IdentityIdentityconv1d_82/mul_1:z:0*
T0*+
_output_shapes
:џџџџџџџџџaь
conv1d_82/IdentityN	IdentityNconv1d_82/mul_1:z:0conv1d_82/BiasAdd:output:0conv1d_82/beta:output:0*
T
2*,
_gradient_op_typeCustomGradient-799578*D
_output_shapes2
0:џџџџџџџџџa:џџџџџџџџџa: j
conv1d_83/Conv1D/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
§џџџџџџџџЋ
conv1d_83/Conv1D/ExpandDims
ExpandDimsconv1d_82/IdentityN:output:0(conv1d_83/Conv1D/ExpandDims/dim:output:0*
T0*/
_output_shapes
:џџџџџџџџџaІ
,conv1d_83/Conv1D/ExpandDims_1/ReadVariableOpReadVariableOp5conv1d_83_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:sak*
dtype0c
!conv1d_83/Conv1D/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : О
conv1d_83/Conv1D/ExpandDims_1
ExpandDims4conv1d_83/Conv1D/ExpandDims_1/ReadVariableOp:value:0*conv1d_83/Conv1D/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:sakЪ
conv1d_83/Conv1DConv2D$conv1d_83/Conv1D/ExpandDims:output:0&conv1d_83/Conv1D/ExpandDims_1:output:0*
T0*/
_output_shapes
:џџџџџџџџџk*
paddingSAME*
strides
C
conv1d_83/Conv1D/SqueezeSqueezeconv1d_83/Conv1D:output:0*
T0*+
_output_shapes
:џџџџџџџџџk*
squeeze_dims

§џџџџџџџџ
 conv1d_83/BiasAdd/ReadVariableOpReadVariableOp)conv1d_83_biasadd_readvariableop_resource*
_output_shapes
:k*
dtype0
conv1d_83/BiasAddBiasAdd!conv1d_83/Conv1D/Squeeze:output:0(conv1d_83/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:џџџџџџџџџkS
conv1d_83/betaConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
conv1d_83/mulMulconv1d_83/beta:output:0conv1d_83/BiasAdd:output:0*
T0*+
_output_shapes
:џџџџџџџџџke
conv1d_83/SigmoidSigmoidconv1d_83/mul:z:0*
T0*+
_output_shapes
:џџџџџџџџџk
conv1d_83/mul_1Mulconv1d_83/BiasAdd:output:0conv1d_83/Sigmoid:y:0*
T0*+
_output_shapes
:џџџџџџџџџki
conv1d_83/IdentityIdentityconv1d_83/mul_1:z:0*
T0*+
_output_shapes
:џџџџџџџџџkь
conv1d_83/IdentityN	IdentityNconv1d_83/mul_1:z:0conv1d_83/BiasAdd:output:0conv1d_83/beta:output:0*
T
2*,
_gradient_op_typeCustomGradient-799598*D
_output_shapes2
0:џџџџџџџџџk:џџџџџџџџџk: a
flatten_76/ConstConst*
_output_shapes
:*
dtype0*
valueB"џџџџk   
flatten_76/ReshapeReshapeconv1d_83/IdentityN:output:0flatten_76/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџk
%INJECTION_MASKS/MatMul/ReadVariableOpReadVariableOp.injection_masks_matmul_readvariableop_resource*
_output_shapes

:k*
dtype0
INJECTION_MASKS/MatMulMatMulflatten_76/Reshape:output:0-INJECTION_MASKS/MatMul/ReadVariableOp:value:0*
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
:џџџџџџџџџг
NoOpNoOp'^INJECTION_MASKS/BiasAdd/ReadVariableOp&^INJECTION_MASKS/MatMul/ReadVariableOp!^conv1d_81/BiasAdd/ReadVariableOp-^conv1d_81/Conv1D/ExpandDims_1/ReadVariableOp!^conv1d_82/BiasAdd/ReadVariableOp-^conv1d_82/Conv1D/ExpandDims_1/ReadVariableOp!^conv1d_83/BiasAdd/ReadVariableOp-^conv1d_83/Conv1D/ExpandDims_1/ReadVariableOp ^dense_89/BiasAdd/ReadVariableOp"^dense_89/Tensordot/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*X
_input_shapesG
E:џџџџџџџџџ:џџџџџџџџџ : : : : : : : : : : 2P
&INJECTION_MASKS/BiasAdd/ReadVariableOp&INJECTION_MASKS/BiasAdd/ReadVariableOp2N
%INJECTION_MASKS/MatMul/ReadVariableOp%INJECTION_MASKS/MatMul/ReadVariableOp2D
 conv1d_81/BiasAdd/ReadVariableOp conv1d_81/BiasAdd/ReadVariableOp2\
,conv1d_81/Conv1D/ExpandDims_1/ReadVariableOp,conv1d_81/Conv1D/ExpandDims_1/ReadVariableOp2D
 conv1d_82/BiasAdd/ReadVariableOp conv1d_82/BiasAdd/ReadVariableOp2\
,conv1d_82/Conv1D/ExpandDims_1/ReadVariableOp,conv1d_82/Conv1D/ExpandDims_1/ReadVariableOp2D
 conv1d_83/BiasAdd/ReadVariableOp conv1d_83/BiasAdd/ReadVariableOp2\
,conv1d_83/Conv1D/ExpandDims_1/ReadVariableOp,conv1d_83/Conv1D/ExpandDims_1/ReadVariableOp2B
dense_89/BiasAdd/ReadVariableOpdense_89/BiasAdd/ReadVariableOp2F
!dense_89/Tensordot/ReadVariableOp!dense_89/Tensordot/ReadVariableOp:]Y
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
Ќ
K
#__inference__update_step_xla_286511
gradient
variable:*
_XlaMustCompile(*(
_construction_contextkEagerRuntime*
_input_shapes

:: *
	_noinline(:($
"
_user_specified_name
variable:D @

_output_shapes
:
"
_user_specified_name
gradient
 
E
)__inference_restored_function_body_285673

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
C__inference_reshape_76_layer_call_and_return_conditional_losses_534e
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
р
В
#__inference_internal_grad_fn_800117
result_grads_0
result_grads_1
result_grads_2
mul_conv1d_82_beta
mul_conv1d_82_biasadd
identity

identity_1|
mulMulmul_conv1d_82_betamul_conv1d_82_biasadd^result_grads_0*
T0*+
_output_shapes
:џџџџџџџџџaQ
SigmoidSigmoidmul:z:0*
T0*+
_output_shapes
:џџџџџџџџџam
mul_1Mulmul_conv1d_82_betamul_conv1d_82_biasadd*
T0*+
_output_shapes
:џџџџџџџџџaJ
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?]
subSubsub/x:output:0Sigmoid:y:0*
T0*+
_output_shapes
:џџџџџџџџџaV
mul_2Mul	mul_1:z:0sub:z:0*
T0*+
_output_shapes
:џџџџџџџџџaJ
add/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?]
addAddV2add/x:output:0	mul_2:z:0*
T0*+
_output_shapes
:џџџџџџџџџaX
mul_3MulSigmoid:y:0add:z:0*
T0*+
_output_shapes
:џџџџџџџџџa]
SquareSquaremul_conv1d_82_biasadd*
T0*+
_output_shapes
:џџџџџџџџџa^
mul_4Mulresult_grads_0
Square:y:0*
T0*+
_output_shapes
:џџџџџџџџџaZ
mul_5Mul	mul_4:z:0Sigmoid:y:0*
T0*+
_output_shapes
:џџџџџџџџџaL
sub_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?a
sub_1Subsub_1/x:output:0Sigmoid:y:0*
T0*+
_output_shapes
:џџџџџџџџџaX
mul_6Mul	mul_5:z:0	sub_1:z:0*
T0*+
_output_shapes
:џџџџџџџџџaZ
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
:џџџџџџџџџaU
IdentityIdentity	mul_7:z:0*
T0*+
_output_shapes
:џџџџџџџџџaE

Identity_1IdentitySum:output:0*
T0*
_output_shapes
: "!

identity_1Identity_1:output:0"
identityIdentity:output:0*\
_input_shapesK
I:џџџџџџџџџa:џџџџџџџџџa: : :џџџџџџџџџa:1-
+
_output_shapes
:џџџџџџџџџa:
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
:џџџџџџџџџa
(
_user_specified_nameresult_grads_1: 
&
 _has_manual_control_dependencies(
+
_output_shapes
:џџџџџџџџџa
(
_user_specified_nameresult_grads_0
Ы

)__inference_model_76_layer_call_fn_799312
	offsource
onsource
unknown:Fy
	unknown_0:y
	unknown_1:y
	unknown_2:
	unknown_3:ja
	unknown_4:a
	unknown_5:sak
	unknown_6:k
	unknown_7:k
	unknown_8:
identityЂStatefulPartitionedCallо
StatefulPartitionedCallStatefulPartitionedCall	offsourceonsourceunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*,
_read_only_resource_inputs

	
*<
config_proto,*

CPU

GPU(2*0J

  zE8 *M
fHRF
D__inference_model_76_layer_call_and_return_conditional_losses_799289o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*X
_input_shapesG
E:џџџџџџџџџ:џџџџџџџџџ : : : : : : : : : : 22
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


#__inference_internal_grad_fn_800005
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
:џџџџџџџџџkQ
SigmoidSigmoidmul:z:0*
T0*+
_output_shapes
:џџџџџџџџџkY
mul_1Mulmul_betamul_biasadd*
T0*+
_output_shapes
:џџџџџџџџџkJ
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?]
subSubsub/x:output:0Sigmoid:y:0*
T0*+
_output_shapes
:џџџџџџџџџkV
mul_2Mul	mul_1:z:0sub:z:0*
T0*+
_output_shapes
:џџџџџџџџџkJ
add/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?]
addAddV2add/x:output:0	mul_2:z:0*
T0*+
_output_shapes
:џџџџџџџџџkX
mul_3MulSigmoid:y:0add:z:0*
T0*+
_output_shapes
:џџџџџџџџџkS
SquareSquaremul_biasadd*
T0*+
_output_shapes
:џџџџџџџџџk^
mul_4Mulresult_grads_0
Square:y:0*
T0*+
_output_shapes
:џџџџџџџџџkZ
mul_5Mul	mul_4:z:0Sigmoid:y:0*
T0*+
_output_shapes
:џџџџџџџџџkL
sub_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?a
sub_1Subsub_1/x:output:0Sigmoid:y:0*
T0*+
_output_shapes
:џџџџџџџџџkX
mul_6Mul	mul_5:z:0	sub_1:z:0*
T0*+
_output_shapes
:џџџџџџџџџkZ
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
:џџџџџџџџџkU
IdentityIdentity	mul_7:z:0*
T0*+
_output_shapes
:џџџџџџџџџkE

Identity_1IdentitySum:output:0*
T0*
_output_shapes
: "!

identity_1Identity_1:output:0"
identityIdentity:output:0*\
_input_shapesK
I:џџџџџџџџџk:џџџџџџџџџk: : :џџџџџџџџџk:1-
+
_output_shapes
:џџџџџџџџџk:
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
:џџџџџџџџџk
(
_user_specified_nameresult_grads_1: 
&
 _has_manual_control_dependencies(
+
_output_shapes
:џџџџџџџџџk
(
_user_specified_nameresult_grads_0
ё*
Й
D__inference_model_76_layer_call_and_return_conditional_losses_799229
inputs_1

inputs&
conv1d_81_799201:Fy
conv1d_81_799203:y!
dense_89_799207:y
dense_89_799209:&
conv1d_82_799212:ja
conv1d_82_799214:a&
conv1d_83_799217:sak
conv1d_83_799219:k(
injection_masks_799223:k$
injection_masks_799225:
identityЂ'INJECTION_MASKS/StatefulPartitionedCallЂ!conv1d_81/StatefulPartitionedCallЂ!conv1d_82/StatefulPartitionedCallЂ!conv1d_83/StatefulPartitionedCallЂ dense_89/StatefulPartitionedCallЂ"dropout_83/StatefulPartitionedCallР
%whiten_passthrough_38/PartitionedCallPartitionedCallinputs_1*
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
)__inference_restored_function_body_285667л
reshape_76/PartitionedCallPartitionedCall.whiten_passthrough_38/PartitionedCall:output:0*
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
)__inference_restored_function_body_285673Є
!conv1d_81/StatefulPartitionedCallStatefulPartitionedCall#reshape_76/PartitionedCall:output:0conv1d_81_799201conv1d_81_799203*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:џџџџџџџџџ2y*$
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
E__inference_conv1d_81_layer_call_and_return_conditional_losses_799009
"dropout_83/StatefulPartitionedCallStatefulPartitionedCall*conv1d_81/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:џџџџџџџџџ2y* 
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
F__inference_dropout_83_layer_call_and_return_conditional_losses_799027Ј
 dense_89/StatefulPartitionedCallStatefulPartitionedCall+dropout_83/StatefulPartitionedCall:output:0dense_89_799207dense_89_799209*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:џџџџџџџџџ2*$
_read_only_resource_inputs
*<
config_proto,*

CPU

GPU(2*0J

  zE8 *M
fHRF
D__inference_dense_89_layer_call_and_return_conditional_losses_799060Њ
!conv1d_82/StatefulPartitionedCallStatefulPartitionedCall)dense_89/StatefulPartitionedCall:output:0conv1d_82_799212conv1d_82_799214*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:џџџџџџџџџa*$
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
E__inference_conv1d_82_layer_call_and_return_conditional_losses_799090Ћ
!conv1d_83/StatefulPartitionedCallStatefulPartitionedCall*conv1d_82/StatefulPartitionedCall:output:0conv1d_83_799217conv1d_83_799219*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:џџџџџџџџџk*$
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
E__inference_conv1d_83_layer_call_and_return_conditional_losses_799120я
flatten_76/PartitionedCallPartitionedCall*conv1d_83/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџk* 
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
F__inference_flatten_76_layer_call_and_return_conditional_losses_799132И
'INJECTION_MASKS/StatefulPartitionedCallStatefulPartitionedCall#flatten_76/PartitionedCall:output:0injection_masks_799223injection_masks_799225*
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
K__inference_INJECTION_MASKS_layer_call_and_return_conditional_losses_799145
IdentityIdentity0INJECTION_MASKS/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџЄ
NoOpNoOp(^INJECTION_MASKS/StatefulPartitionedCall"^conv1d_81/StatefulPartitionedCall"^conv1d_82/StatefulPartitionedCall"^conv1d_83/StatefulPartitionedCall!^dense_89/StatefulPartitionedCall#^dropout_83/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*X
_input_shapesG
E:џџџџџџџџџ:џџџџџџџџџ : : : : : : : : : : 2R
'INJECTION_MASKS/StatefulPartitionedCall'INJECTION_MASKS/StatefulPartitionedCall2F
!conv1d_81/StatefulPartitionedCall!conv1d_81/StatefulPartitionedCall2F
!conv1d_82/StatefulPartitionedCall!conv1d_82/StatefulPartitionedCall2F
!conv1d_83/StatefulPartitionedCall!conv1d_83/StatefulPartitionedCall2D
 dense_89/StatefulPartitionedCall dense_89/StatefulPartitionedCall2H
"dropout_83/StatefulPartitionedCall"dropout_83/StatefulPartitionedCall:TP
,
_output_shapes
:џџџџџџџџџ 
 
_user_specified_nameinputs:U Q
-
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
Ѓ

$__inference_signature_wrapper_799461
	offsource
onsource
unknown:Fy
	unknown_0:y
	unknown_1:y
	unknown_2:
	unknown_3:ja
	unknown_4:a
	unknown_5:sak
	unknown_6:k
	unknown_7:k
	unknown_8:
identityЂStatefulPartitionedCallЛ
StatefulPartitionedCallStatefulPartitionedCall	offsourceonsourceunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*,
_read_only_resource_inputs

	
*<
config_proto,*

CPU

GPU(2*0J

  zE8 **
f%R#
!__inference__wrapped_model_798986o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*X
_input_shapesG
E:џџџџџџџџџ:џџџџџџџџџ : : : : : : : : : : 22
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
њ*
М
D__inference_model_76_layer_call_and_return_conditional_losses_799152
	offsource
onsource&
conv1d_81_799010:Fy
conv1d_81_799012:y!
dense_89_799061:y
dense_89_799063:&
conv1d_82_799091:ja
conv1d_82_799093:a&
conv1d_83_799121:sak
conv1d_83_799123:k(
injection_masks_799146:k$
injection_masks_799148:
identityЂ'INJECTION_MASKS/StatefulPartitionedCallЂ!conv1d_81/StatefulPartitionedCallЂ!conv1d_82/StatefulPartitionedCallЂ!conv1d_83/StatefulPartitionedCallЂ dense_89/StatefulPartitionedCallЂ"dropout_83/StatefulPartitionedCallС
%whiten_passthrough_38/PartitionedCallPartitionedCall	offsource*
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
)__inference_restored_function_body_285667л
reshape_76/PartitionedCallPartitionedCall.whiten_passthrough_38/PartitionedCall:output:0*
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
)__inference_restored_function_body_285673Є
!conv1d_81/StatefulPartitionedCallStatefulPartitionedCall#reshape_76/PartitionedCall:output:0conv1d_81_799010conv1d_81_799012*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:џџџџџџџџџ2y*$
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
E__inference_conv1d_81_layer_call_and_return_conditional_losses_799009
"dropout_83/StatefulPartitionedCallStatefulPartitionedCall*conv1d_81/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:џџџџџџџџџ2y* 
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
F__inference_dropout_83_layer_call_and_return_conditional_losses_799027Ј
 dense_89/StatefulPartitionedCallStatefulPartitionedCall+dropout_83/StatefulPartitionedCall:output:0dense_89_799061dense_89_799063*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:џџџџџџџџџ2*$
_read_only_resource_inputs
*<
config_proto,*

CPU

GPU(2*0J

  zE8 *M
fHRF
D__inference_dense_89_layer_call_and_return_conditional_losses_799060Њ
!conv1d_82/StatefulPartitionedCallStatefulPartitionedCall)dense_89/StatefulPartitionedCall:output:0conv1d_82_799091conv1d_82_799093*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:џџџџџџџџџa*$
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
E__inference_conv1d_82_layer_call_and_return_conditional_losses_799090Ћ
!conv1d_83/StatefulPartitionedCallStatefulPartitionedCall*conv1d_82/StatefulPartitionedCall:output:0conv1d_83_799121conv1d_83_799123*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:џџџџџџџџџk*$
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
E__inference_conv1d_83_layer_call_and_return_conditional_losses_799120я
flatten_76/PartitionedCallPartitionedCall*conv1d_83/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџk* 
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
F__inference_flatten_76_layer_call_and_return_conditional_losses_799132И
'INJECTION_MASKS/StatefulPartitionedCallStatefulPartitionedCall#flatten_76/PartitionedCall:output:0injection_masks_799146injection_masks_799148*
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
K__inference_INJECTION_MASKS_layer_call_and_return_conditional_losses_799145
IdentityIdentity0INJECTION_MASKS/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџЄ
NoOpNoOp(^INJECTION_MASKS/StatefulPartitionedCall"^conv1d_81/StatefulPartitionedCall"^conv1d_82/StatefulPartitionedCall"^conv1d_83/StatefulPartitionedCall!^dense_89/StatefulPartitionedCall#^dropout_83/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*X
_input_shapesG
E:џџџџџџџџџ:џџџџџџџџџ : : : : : : : : : : 2R
'INJECTION_MASKS/StatefulPartitionedCall'INJECTION_MASKS/StatefulPartitionedCall2F
!conv1d_81/StatefulPartitionedCall!conv1d_81/StatefulPartitionedCall2F
!conv1d_82/StatefulPartitionedCall!conv1d_82/StatefulPartitionedCall2F
!conv1d_83/StatefulPartitionedCall!conv1d_83/StatefulPartitionedCall2D
 dense_89/StatefulPartitionedCall dense_89/StatefulPartitionedCall2H
"dropout_83/StatefulPartitionedCall"dropout_83/StatefulPartitionedCall:VR
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
К
Ё
"__inference__traced_restore_800539
file_prefix/
assignvariableop_kernel_4:Fy'
assignvariableop_1_bias_4:y-
assignvariableop_2_kernel_3:y'
assignvariableop_3_bias_3:1
assignvariableop_4_kernel_2:ja'
assignvariableop_5_bias_2:a1
assignvariableop_6_kernel_1:sak'
assignvariableop_7_bias_1:k+
assignvariableop_8_kernel:k%
assignvariableop_9_bias:'
assignvariableop_10_iteration:	 +
!assignvariableop_11_learning_rate: 9
#assignvariableop_12_adam_m_kernel_4:Fy9
#assignvariableop_13_adam_v_kernel_4:Fy/
!assignvariableop_14_adam_m_bias_4:y/
!assignvariableop_15_adam_v_bias_4:y5
#assignvariableop_16_adam_m_kernel_3:y5
#assignvariableop_17_adam_v_kernel_3:y/
!assignvariableop_18_adam_m_bias_3:/
!assignvariableop_19_adam_v_bias_3:9
#assignvariableop_20_adam_m_kernel_2:ja9
#assignvariableop_21_adam_v_kernel_2:ja/
!assignvariableop_22_adam_m_bias_2:a/
!assignvariableop_23_adam_v_bias_2:a9
#assignvariableop_24_adam_m_kernel_1:sak9
#assignvariableop_25_adam_v_kernel_1:sak/
!assignvariableop_26_adam_m_bias_1:k/
!assignvariableop_27_adam_v_bias_1:k3
!assignvariableop_28_adam_m_kernel:k3
!assignvariableop_29_adam_v_kernel:k-
assignvariableop_30_adam_m_bias:-
assignvariableop_31_adam_v_bias:%
assignvariableop_32_total_1: %
assignvariableop_33_count_1: #
assignvariableop_34_total: #
assignvariableop_35_count: 
identity_37ЂAssignVariableOpЂAssignVariableOp_1ЂAssignVariableOp_10ЂAssignVariableOp_11ЂAssignVariableOp_12ЂAssignVariableOp_13ЂAssignVariableOp_14ЂAssignVariableOp_15ЂAssignVariableOp_16ЂAssignVariableOp_17ЂAssignVariableOp_18ЂAssignVariableOp_19ЂAssignVariableOp_2ЂAssignVariableOp_20ЂAssignVariableOp_21ЂAssignVariableOp_22ЂAssignVariableOp_23ЂAssignVariableOp_24ЂAssignVariableOp_25ЂAssignVariableOp_26ЂAssignVariableOp_27ЂAssignVariableOp_28ЂAssignVariableOp_29ЂAssignVariableOp_3ЂAssignVariableOp_30ЂAssignVariableOp_31ЂAssignVariableOp_32ЂAssignVariableOp_33ЂAssignVariableOp_34ЂAssignVariableOp_35ЂAssignVariableOp_4ЂAssignVariableOp_5ЂAssignVariableOp_6ЂAssignVariableOp_7ЂAssignVariableOp_8ЂAssignVariableOp_9љ
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:%*
dtype0*
valueB%B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB0optimizer/_iterations/.ATTRIBUTES/VARIABLE_VALUEB3optimizer/_learning_rate/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/1/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/2/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/3/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/4/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/5/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/6/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/7/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/8/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/9/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/10/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/11/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/12/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/13/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/14/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/15/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/16/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/17/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/18/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/19/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/20/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPHК
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:%*
dtype0*]
valueTBR%B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B к
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*Њ
_output_shapes
:::::::::::::::::::::::::::::::::::::*3
dtypes)
'2%	[
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:Ќ
AssignVariableOpAssignVariableOpassignvariableop_kernel_4Identity:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:А
AssignVariableOp_1AssignVariableOpassignvariableop_1_bias_4Identity_1:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:В
AssignVariableOp_2AssignVariableOpassignvariableop_2_kernel_3Identity_2:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:А
AssignVariableOp_3AssignVariableOpassignvariableop_3_bias_3Identity_3:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:В
AssignVariableOp_4AssignVariableOpassignvariableop_4_kernel_2Identity_4:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:А
AssignVariableOp_5AssignVariableOpassignvariableop_5_bias_2Identity_5:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:В
AssignVariableOp_6AssignVariableOpassignvariableop_6_kernel_1Identity_6:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:А
AssignVariableOp_7AssignVariableOpassignvariableop_7_bias_1Identity_7:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:А
AssignVariableOp_8AssignVariableOpassignvariableop_8_kernelIdentity_8:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:Ў
AssignVariableOp_9AssignVariableOpassignvariableop_9_biasIdentity_9:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0	*
_output_shapes
:Ж
AssignVariableOp_10AssignVariableOpassignvariableop_10_iterationIdentity_10:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0	_
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:К
AssignVariableOp_11AssignVariableOp!assignvariableop_11_learning_rateIdentity_11:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:М
AssignVariableOp_12AssignVariableOp#assignvariableop_12_adam_m_kernel_4Identity_12:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:М
AssignVariableOp_13AssignVariableOp#assignvariableop_13_adam_v_kernel_4Identity_13:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:К
AssignVariableOp_14AssignVariableOp!assignvariableop_14_adam_m_bias_4Identity_14:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:К
AssignVariableOp_15AssignVariableOp!assignvariableop_15_adam_v_bias_4Identity_15:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:М
AssignVariableOp_16AssignVariableOp#assignvariableop_16_adam_m_kernel_3Identity_16:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:М
AssignVariableOp_17AssignVariableOp#assignvariableop_17_adam_v_kernel_3Identity_17:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:К
AssignVariableOp_18AssignVariableOp!assignvariableop_18_adam_m_bias_3Identity_18:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:К
AssignVariableOp_19AssignVariableOp!assignvariableop_19_adam_v_bias_3Identity_19:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:М
AssignVariableOp_20AssignVariableOp#assignvariableop_20_adam_m_kernel_2Identity_20:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:М
AssignVariableOp_21AssignVariableOp#assignvariableop_21_adam_v_kernel_2Identity_21:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:К
AssignVariableOp_22AssignVariableOp!assignvariableop_22_adam_m_bias_2Identity_22:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:К
AssignVariableOp_23AssignVariableOp!assignvariableop_23_adam_v_bias_2Identity_23:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:М
AssignVariableOp_24AssignVariableOp#assignvariableop_24_adam_m_kernel_1Identity_24:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:М
AssignVariableOp_25AssignVariableOp#assignvariableop_25_adam_v_kernel_1Identity_25:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:К
AssignVariableOp_26AssignVariableOp!assignvariableop_26_adam_m_bias_1Identity_26:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:К
AssignVariableOp_27AssignVariableOp!assignvariableop_27_adam_v_bias_1Identity_27:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:К
AssignVariableOp_28AssignVariableOp!assignvariableop_28_adam_m_kernelIdentity_28:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:К
AssignVariableOp_29AssignVariableOp!assignvariableop_29_adam_v_kernelIdentity_29:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:И
AssignVariableOp_30AssignVariableOpassignvariableop_30_adam_m_biasIdentity_30:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:И
AssignVariableOp_31AssignVariableOpassignvariableop_31_adam_v_biasIdentity_31:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:Д
AssignVariableOp_32AssignVariableOpassignvariableop_32_total_1Identity_32:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:Д
AssignVariableOp_33AssignVariableOpassignvariableop_33_count_1Identity_33:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_34IdentityRestoreV2:tensors:34"/device:CPU:0*
T0*
_output_shapes
:В
AssignVariableOp_34AssignVariableOpassignvariableop_34_totalIdentity_34:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_35IdentityRestoreV2:tensors:35"/device:CPU:0*
T0*
_output_shapes
:В
AssignVariableOp_35AssignVariableOpassignvariableop_35_countIdentity_35:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0Y
NoOpNoOp"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 ч
Identity_36Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: W
Identity_37IdentityIdentity_36:output:0^NoOp_1*
T0*
_output_shapes
: д
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*"
_acd_function_control_output(*
_output_shapes
 "#
identity_37Identity_37:output:0*]
_input_shapesL
J: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_16AssignVariableOp_162*
AssignVariableOp_17AssignVariableOp_172*
AssignVariableOp_18AssignVariableOp_182*
AssignVariableOp_19AssignVariableOp_192(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_20AssignVariableOp_202*
AssignVariableOp_21AssignVariableOp_212*
AssignVariableOp_22AssignVariableOp_222*
AssignVariableOp_23AssignVariableOp_232*
AssignVariableOp_24AssignVariableOp_242*
AssignVariableOp_25AssignVariableOp_252*
AssignVariableOp_26AssignVariableOp_262*
AssignVariableOp_27AssignVariableOp_272*
AssignVariableOp_28AssignVariableOp_282*
AssignVariableOp_29AssignVariableOp_292(
AssignVariableOp_2AssignVariableOp_22*
AssignVariableOp_30AssignVariableOp_302*
AssignVariableOp_31AssignVariableOp_312*
AssignVariableOp_32AssignVariableOp_322*
AssignVariableOp_33AssignVariableOp_332*
AssignVariableOp_34AssignVariableOp_342*
AssignVariableOp_35AssignVariableOp_352(
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
Ф
S
#__inference__update_step_xla_286526
gradient
variable:sak*
_XlaMustCompile(*(
_construction_contextkEagerRuntime*#
_input_shapes
:sak: *
	_noinline(:($
"
_user_specified_name
variable:L H
"
_output_shapes
:sak
"
_user_specified_name
gradient
Ы

)__inference_model_76_layer_call_fn_799252
	offsource
onsource
unknown:Fy
	unknown_0:y
	unknown_1:y
	unknown_2:
	unknown_3:ja
	unknown_4:a
	unknown_5:sak
	unknown_6:k
	unknown_7:k
	unknown_8:
identityЂStatefulPartitionedCallо
StatefulPartitionedCallStatefulPartitionedCall	offsourceonsourceunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*,
_read_only_resource_inputs

	
*<
config_proto,*

CPU

GPU(2*0J

  zE8 *M
fHRF
D__inference_model_76_layer_call_and_return_conditional_losses_799229o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*X
_input_shapesG
E:џџџџџџџџџ:џџџџџџџџџ : : : : : : : : : : 22
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
n
	
D__inference_model_76_layer_call_and_return_conditional_losses_799712
inputs_offsource
inputs_onsourceK
5conv1d_81_conv1d_expanddims_1_readvariableop_resource:Fy7
)conv1d_81_biasadd_readvariableop_resource:y<
*dense_89_tensordot_readvariableop_resource:y6
(dense_89_biasadd_readvariableop_resource:K
5conv1d_82_conv1d_expanddims_1_readvariableop_resource:ja7
)conv1d_82_biasadd_readvariableop_resource:aK
5conv1d_83_conv1d_expanddims_1_readvariableop_resource:sak7
)conv1d_83_biasadd_readvariableop_resource:k@
.injection_masks_matmul_readvariableop_resource:k=
/injection_masks_biasadd_readvariableop_resource:
identityЂ&INJECTION_MASKS/BiasAdd/ReadVariableOpЂ%INJECTION_MASKS/MatMul/ReadVariableOpЂ conv1d_81/BiasAdd/ReadVariableOpЂ,conv1d_81/Conv1D/ExpandDims_1/ReadVariableOpЂ conv1d_82/BiasAdd/ReadVariableOpЂ,conv1d_82/Conv1D/ExpandDims_1/ReadVariableOpЂ conv1d_83/BiasAdd/ReadVariableOpЂ,conv1d_83/Conv1D/ExpandDims_1/ReadVariableOpЂdense_89/BiasAdd/ReadVariableOpЂ!dense_89/Tensordot/ReadVariableOpШ
%whiten_passthrough_38/PartitionedCallPartitionedCallinputs_offsource*
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
)__inference_restored_function_body_285667л
reshape_76/PartitionedCallPartitionedCall.whiten_passthrough_38/PartitionedCall:output:0*
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
)__inference_restored_function_body_285673j
conv1d_81/Conv1D/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
§џџџџџџџџГ
conv1d_81/Conv1D/ExpandDims
ExpandDims#reshape_76/PartitionedCall:output:0(conv1d_81/Conv1D/ExpandDims/dim:output:0*
T0*0
_output_shapes
:џџџџџџџџџІ
,conv1d_81/Conv1D/ExpandDims_1/ReadVariableOpReadVariableOp5conv1d_81_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:Fy*
dtype0c
!conv1d_81/Conv1D/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : О
conv1d_81/Conv1D/ExpandDims_1
ExpandDims4conv1d_81/Conv1D/ExpandDims_1/ReadVariableOp:value:0*conv1d_81/Conv1D/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:FyЪ
conv1d_81/Conv1DConv2D$conv1d_81/Conv1D/ExpandDims:output:0&conv1d_81/Conv1D/ExpandDims_1:output:0*
T0*/
_output_shapes
:џџџџџџџџџ2y*
paddingSAME*
strides
)
conv1d_81/Conv1D/SqueezeSqueezeconv1d_81/Conv1D:output:0*
T0*+
_output_shapes
:џџџџџџџџџ2y*
squeeze_dims

§џџџџџџџџ
 conv1d_81/BiasAdd/ReadVariableOpReadVariableOp)conv1d_81_biasadd_readvariableop_resource*
_output_shapes
:y*
dtype0
conv1d_81/BiasAddBiasAdd!conv1d_81/Conv1D/Squeeze:output:0(conv1d_81/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:џџџџџџџџџ2yn
conv1d_81/SigmoidSigmoidconv1d_81/BiasAdd:output:0*
T0*+
_output_shapes
:џџџџџџџџџ2yl
dropout_83/IdentityIdentityconv1d_81/Sigmoid:y:0*
T0*+
_output_shapes
:џџџџџџџџџ2y
!dense_89/Tensordot/ReadVariableOpReadVariableOp*dense_89_tensordot_readvariableop_resource*
_output_shapes

:y*
dtype0a
dense_89/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:h
dense_89/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       r
dense_89/Tensordot/ShapeShapedropout_83/Identity:output:0*
T0*
_output_shapes
::эЯb
 dense_89/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : п
dense_89/Tensordot/GatherV2GatherV2!dense_89/Tensordot/Shape:output:0 dense_89/Tensordot/free:output:0)dense_89/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:d
"dense_89/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : у
dense_89/Tensordot/GatherV2_1GatherV2!dense_89/Tensordot/Shape:output:0 dense_89/Tensordot/axes:output:0+dense_89/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:b
dense_89/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 
dense_89/Tensordot/ProdProd$dense_89/Tensordot/GatherV2:output:0!dense_89/Tensordot/Const:output:0*
T0*
_output_shapes
: d
dense_89/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 
dense_89/Tensordot/Prod_1Prod&dense_89/Tensordot/GatherV2_1:output:0#dense_89/Tensordot/Const_1:output:0*
T0*
_output_shapes
: `
dense_89/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : Р
dense_89/Tensordot/concatConcatV2 dense_89/Tensordot/free:output:0 dense_89/Tensordot/axes:output:0'dense_89/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:
dense_89/Tensordot/stackPack dense_89/Tensordot/Prod:output:0"dense_89/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:Ё
dense_89/Tensordot/transpose	Transposedropout_83/Identity:output:0"dense_89/Tensordot/concat:output:0*
T0*+
_output_shapes
:џџџџџџџџџ2yЅ
dense_89/Tensordot/ReshapeReshape dense_89/Tensordot/transpose:y:0!dense_89/Tensordot/stack:output:0*
T0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџЅ
dense_89/Tensordot/MatMulMatMul#dense_89/Tensordot/Reshape:output:0)dense_89/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџd
dense_89/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:b
 dense_89/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : Ы
dense_89/Tensordot/concat_1ConcatV2$dense_89/Tensordot/GatherV2:output:0#dense_89/Tensordot/Const_2:output:0)dense_89/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:
dense_89/TensordotReshape#dense_89/Tensordot/MatMul:product:0$dense_89/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:џџџџџџџџџ2
dense_89/BiasAdd/ReadVariableOpReadVariableOp(dense_89_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
dense_89/BiasAddBiasAdddense_89/Tensordot:output:0'dense_89/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:џџџџџџџџџ2l
dense_89/SigmoidSigmoiddense_89/BiasAdd:output:0*
T0*+
_output_shapes
:џџџџџџџџџ2j
conv1d_82/Conv1D/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
§џџџџџџџџЃ
conv1d_82/Conv1D/ExpandDims
ExpandDimsdense_89/Sigmoid:y:0(conv1d_82/Conv1D/ExpandDims/dim:output:0*
T0*/
_output_shapes
:џџџџџџџџџ2І
,conv1d_82/Conv1D/ExpandDims_1/ReadVariableOpReadVariableOp5conv1d_82_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:ja*
dtype0c
!conv1d_82/Conv1D/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : О
conv1d_82/Conv1D/ExpandDims_1
ExpandDims4conv1d_82/Conv1D/ExpandDims_1/ReadVariableOp:value:0*conv1d_82/Conv1D/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:jaЪ
conv1d_82/Conv1DConv2D$conv1d_82/Conv1D/ExpandDims:output:0&conv1d_82/Conv1D/ExpandDims_1:output:0*
T0*/
_output_shapes
:џџџџџџџџџa*
paddingSAME*
strides
=
conv1d_82/Conv1D/SqueezeSqueezeconv1d_82/Conv1D:output:0*
T0*+
_output_shapes
:џџџџџџџџџa*
squeeze_dims

§џџџџџџџџ
 conv1d_82/BiasAdd/ReadVariableOpReadVariableOp)conv1d_82_biasadd_readvariableop_resource*
_output_shapes
:a*
dtype0
conv1d_82/BiasAddBiasAdd!conv1d_82/Conv1D/Squeeze:output:0(conv1d_82/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:џџџџџџџџџaS
conv1d_82/betaConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
conv1d_82/mulMulconv1d_82/beta:output:0conv1d_82/BiasAdd:output:0*
T0*+
_output_shapes
:џџџџџџџџџae
conv1d_82/SigmoidSigmoidconv1d_82/mul:z:0*
T0*+
_output_shapes
:џџџџџџџџџa
conv1d_82/mul_1Mulconv1d_82/BiasAdd:output:0conv1d_82/Sigmoid:y:0*
T0*+
_output_shapes
:џџџџџџџџџai
conv1d_82/IdentityIdentityconv1d_82/mul_1:z:0*
T0*+
_output_shapes
:џџџџџџџџџaь
conv1d_82/IdentityN	IdentityNconv1d_82/mul_1:z:0conv1d_82/BiasAdd:output:0conv1d_82/beta:output:0*
T
2*,
_gradient_op_typeCustomGradient-799674*D
_output_shapes2
0:џџџџџџџџџa:џџџџџџџџџa: j
conv1d_83/Conv1D/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
§џџџџџџџџЋ
conv1d_83/Conv1D/ExpandDims
ExpandDimsconv1d_82/IdentityN:output:0(conv1d_83/Conv1D/ExpandDims/dim:output:0*
T0*/
_output_shapes
:џџџџџџџџџaІ
,conv1d_83/Conv1D/ExpandDims_1/ReadVariableOpReadVariableOp5conv1d_83_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:sak*
dtype0c
!conv1d_83/Conv1D/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : О
conv1d_83/Conv1D/ExpandDims_1
ExpandDims4conv1d_83/Conv1D/ExpandDims_1/ReadVariableOp:value:0*conv1d_83/Conv1D/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:sakЪ
conv1d_83/Conv1DConv2D$conv1d_83/Conv1D/ExpandDims:output:0&conv1d_83/Conv1D/ExpandDims_1:output:0*
T0*/
_output_shapes
:џџџџџџџџџk*
paddingSAME*
strides
C
conv1d_83/Conv1D/SqueezeSqueezeconv1d_83/Conv1D:output:0*
T0*+
_output_shapes
:џџџџџџџџџk*
squeeze_dims

§џџџџџџџџ
 conv1d_83/BiasAdd/ReadVariableOpReadVariableOp)conv1d_83_biasadd_readvariableop_resource*
_output_shapes
:k*
dtype0
conv1d_83/BiasAddBiasAdd!conv1d_83/Conv1D/Squeeze:output:0(conv1d_83/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:џџџџџџџџџkS
conv1d_83/betaConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
conv1d_83/mulMulconv1d_83/beta:output:0conv1d_83/BiasAdd:output:0*
T0*+
_output_shapes
:џџџџџџџџџke
conv1d_83/SigmoidSigmoidconv1d_83/mul:z:0*
T0*+
_output_shapes
:џџџџџџџџџk
conv1d_83/mul_1Mulconv1d_83/BiasAdd:output:0conv1d_83/Sigmoid:y:0*
T0*+
_output_shapes
:џџџџџџџџџki
conv1d_83/IdentityIdentityconv1d_83/mul_1:z:0*
T0*+
_output_shapes
:џџџџџџџџџkь
conv1d_83/IdentityN	IdentityNconv1d_83/mul_1:z:0conv1d_83/BiasAdd:output:0conv1d_83/beta:output:0*
T
2*,
_gradient_op_typeCustomGradient-799694*D
_output_shapes2
0:џџџџџџџџџk:џџџџџџџџџk: a
flatten_76/ConstConst*
_output_shapes
:*
dtype0*
valueB"џџџџk   
flatten_76/ReshapeReshapeconv1d_83/IdentityN:output:0flatten_76/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџk
%INJECTION_MASKS/MatMul/ReadVariableOpReadVariableOp.injection_masks_matmul_readvariableop_resource*
_output_shapes

:k*
dtype0
INJECTION_MASKS/MatMulMatMulflatten_76/Reshape:output:0-INJECTION_MASKS/MatMul/ReadVariableOp:value:0*
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
:џџџџџџџџџг
NoOpNoOp'^INJECTION_MASKS/BiasAdd/ReadVariableOp&^INJECTION_MASKS/MatMul/ReadVariableOp!^conv1d_81/BiasAdd/ReadVariableOp-^conv1d_81/Conv1D/ExpandDims_1/ReadVariableOp!^conv1d_82/BiasAdd/ReadVariableOp-^conv1d_82/Conv1D/ExpandDims_1/ReadVariableOp!^conv1d_83/BiasAdd/ReadVariableOp-^conv1d_83/Conv1D/ExpandDims_1/ReadVariableOp ^dense_89/BiasAdd/ReadVariableOp"^dense_89/Tensordot/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*X
_input_shapesG
E:џџџџџџџџџ:џџџџџџџџџ : : : : : : : : : : 2P
&INJECTION_MASKS/BiasAdd/ReadVariableOp&INJECTION_MASKS/BiasAdd/ReadVariableOp2N
%INJECTION_MASKS/MatMul/ReadVariableOp%INJECTION_MASKS/MatMul/ReadVariableOp2D
 conv1d_81/BiasAdd/ReadVariableOp conv1d_81/BiasAdd/ReadVariableOp2\
,conv1d_81/Conv1D/ExpandDims_1/ReadVariableOp,conv1d_81/Conv1D/ExpandDims_1/ReadVariableOp2D
 conv1d_82/BiasAdd/ReadVariableOp conv1d_82/BiasAdd/ReadVariableOp2\
,conv1d_82/Conv1D/ExpandDims_1/ReadVariableOp,conv1d_82/Conv1D/ExpandDims_1/ReadVariableOp2D
 conv1d_83/BiasAdd/ReadVariableOp conv1d_83/BiasAdd/ReadVariableOp2\
,conv1d_83/Conv1D/ExpandDims_1/ReadVariableOp,conv1d_83/Conv1D/ExpandDims_1/ReadVariableOp2B
dense_89/BiasAdd/ReadVariableOpdense_89/BiasAdd/ReadVariableOp2F
!dense_89/Tensordot/ReadVariableOp!dense_89/Tensordot/ReadVariableOp:]Y
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
Ќ
K
#__inference__update_step_xla_286501
gradient
variable:y*
_XlaMustCompile(*(
_construction_contextkEagerRuntime*
_input_shapes

:y: *
	_noinline(:($
"
_user_specified_name
variable:D @

_output_shapes
:y
"
_user_specified_name
gradient
Х)

D__inference_model_76_layer_call_and_return_conditional_losses_799289
inputs_1

inputs&
conv1d_81_799261:Fy
conv1d_81_799263:y!
dense_89_799267:y
dense_89_799269:&
conv1d_82_799272:ja
conv1d_82_799274:a&
conv1d_83_799277:sak
conv1d_83_799279:k(
injection_masks_799283:k$
injection_masks_799285:
identityЂ'INJECTION_MASKS/StatefulPartitionedCallЂ!conv1d_81/StatefulPartitionedCallЂ!conv1d_82/StatefulPartitionedCallЂ!conv1d_83/StatefulPartitionedCallЂ dense_89/StatefulPartitionedCallР
%whiten_passthrough_38/PartitionedCallPartitionedCallinputs_1*
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
)__inference_restored_function_body_285667л
reshape_76/PartitionedCallPartitionedCall.whiten_passthrough_38/PartitionedCall:output:0*
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
)__inference_restored_function_body_285673Є
!conv1d_81/StatefulPartitionedCallStatefulPartitionedCall#reshape_76/PartitionedCall:output:0conv1d_81_799261conv1d_81_799263*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:џџџџџџџџџ2y*$
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
E__inference_conv1d_81_layer_call_and_return_conditional_losses_799009ѓ
dropout_83/PartitionedCallPartitionedCall*conv1d_81/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:џџџџџџџџџ2y* 
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
F__inference_dropout_83_layer_call_and_return_conditional_losses_799167 
 dense_89/StatefulPartitionedCallStatefulPartitionedCall#dropout_83/PartitionedCall:output:0dense_89_799267dense_89_799269*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:џџџџџџџџџ2*$
_read_only_resource_inputs
*<
config_proto,*

CPU

GPU(2*0J

  zE8 *M
fHRF
D__inference_dense_89_layer_call_and_return_conditional_losses_799060Њ
!conv1d_82/StatefulPartitionedCallStatefulPartitionedCall)dense_89/StatefulPartitionedCall:output:0conv1d_82_799272conv1d_82_799274*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:џџџџџџџџџa*$
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
E__inference_conv1d_82_layer_call_and_return_conditional_losses_799090Ћ
!conv1d_83/StatefulPartitionedCallStatefulPartitionedCall*conv1d_82/StatefulPartitionedCall:output:0conv1d_83_799277conv1d_83_799279*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:џџџџџџџџџk*$
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
E__inference_conv1d_83_layer_call_and_return_conditional_losses_799120я
flatten_76/PartitionedCallPartitionedCall*conv1d_83/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџk* 
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
F__inference_flatten_76_layer_call_and_return_conditional_losses_799132И
'INJECTION_MASKS/StatefulPartitionedCallStatefulPartitionedCall#flatten_76/PartitionedCall:output:0injection_masks_799283injection_masks_799285*
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
K__inference_INJECTION_MASKS_layer_call_and_return_conditional_losses_799145
IdentityIdentity0INJECTION_MASKS/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџџ
NoOpNoOp(^INJECTION_MASKS/StatefulPartitionedCall"^conv1d_81/StatefulPartitionedCall"^conv1d_82/StatefulPartitionedCall"^conv1d_83/StatefulPartitionedCall!^dense_89/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*X
_input_shapesG
E:џџџџџџџџџ:џџџџџџџџџ : : : : : : : : : : 2R
'INJECTION_MASKS/StatefulPartitionedCall'INJECTION_MASKS/StatefulPartitionedCall2F
!conv1d_81/StatefulPartitionedCall!conv1d_81/StatefulPartitionedCall2F
!conv1d_82/StatefulPartitionedCall!conv1d_82/StatefulPartitionedCall2F
!conv1d_83/StatefulPartitionedCall!conv1d_83/StatefulPartitionedCall2D
 dense_89/StatefulPartitionedCall dense_89/StatefulPartitionedCall:TP
,
_output_shapes
:џџџџџџџџџ 
 
_user_specified_nameinputs:U Q
-
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
Ё

ќ
K__inference_INJECTION_MASKS_layer_call_and_return_conditional_losses_799145

inputs0
matmul_readvariableop_resource:k-
biasadd_readvariableop_resource:
identityЂBiasAdd/ReadVariableOpЂMatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:k*
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
:џџџџџџџџџk: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:џџџџџџџџџk
 
_user_specified_nameinputs
 
Ф
#__inference_internal_grad_fn_800257
result_grads_0
result_grads_1
result_grads_2
mul_model_76_conv1d_83_beta"
mul_model_76_conv1d_83_biasadd
identity

identity_1
mulMulmul_model_76_conv1d_83_betamul_model_76_conv1d_83_biasadd^result_grads_0*
T0*+
_output_shapes
:џџџџџџџџџkQ
SigmoidSigmoidmul:z:0*
T0*+
_output_shapes
:џџџџџџџџџk
mul_1Mulmul_model_76_conv1d_83_betamul_model_76_conv1d_83_biasadd*
T0*+
_output_shapes
:џџџџџџџџџkJ
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?]
subSubsub/x:output:0Sigmoid:y:0*
T0*+
_output_shapes
:џџџџџџџџџkV
mul_2Mul	mul_1:z:0sub:z:0*
T0*+
_output_shapes
:џџџџџџџџџkJ
add/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?]
addAddV2add/x:output:0	mul_2:z:0*
T0*+
_output_shapes
:џџџџџџџџџkX
mul_3MulSigmoid:y:0add:z:0*
T0*+
_output_shapes
:џџџџџџџџџkf
SquareSquaremul_model_76_conv1d_83_biasadd*
T0*+
_output_shapes
:џџџџџџџџџk^
mul_4Mulresult_grads_0
Square:y:0*
T0*+
_output_shapes
:џџџџџџџџџkZ
mul_5Mul	mul_4:z:0Sigmoid:y:0*
T0*+
_output_shapes
:џџџџџџџџџkL
sub_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?a
sub_1Subsub_1/x:output:0Sigmoid:y:0*
T0*+
_output_shapes
:џџџџџџџџџkX
mul_6Mul	mul_5:z:0	sub_1:z:0*
T0*+
_output_shapes
:џџџџџџџџџkZ
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
:џџџџџџџџџkU
IdentityIdentity	mul_7:z:0*
T0*+
_output_shapes
:џџџџџџџџџkE

Identity_1IdentitySum:output:0*
T0*
_output_shapes
: "!

identity_1Identity_1:output:0"
identityIdentity:output:0*\
_input_shapesK
I:џџџџџџџџџk:џџџџџџџџџk: : :џџџџџџџџџk:1-
+
_output_shapes
:џџџџџџџџџk:
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
:џџџџџџџџџk
(
_user_specified_nameresult_grads_1: 
&
 _has_manual_control_dependencies(
+
_output_shapes
:џџџџџџџџџk
(
_user_specified_nameresult_grads_0
А
ћ
D__inference_dense_89_layer_call_and_return_conditional_losses_799804

inputs3
!tensordot_readvariableop_resource:y-
biasadd_readvariableop_resource:
identityЂBiasAdd/ReadVariableOpЂTensordot/ReadVariableOpz
Tensordot/ReadVariableOpReadVariableOp!tensordot_readvariableop_resource*
_output_shapes

:y*
dtype0X
Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:_
Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       S
Tensordot/ShapeShapeinputs*
T0*
_output_shapes
::эЯY
Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : Л
Tensordot/GatherV2GatherV2Tensordot/Shape:output:0Tensordot/free:output:0 Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:[
Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : П
Tensordot/GatherV2_1GatherV2Tensordot/Shape:output:0Tensordot/axes:output:0"Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:Y
Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: n
Tensordot/ProdProdTensordot/GatherV2:output:0Tensordot/Const:output:0*
T0*
_output_shapes
: [
Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: t
Tensordot/Prod_1ProdTensordot/GatherV2_1:output:0Tensordot/Const_1:output:0*
T0*
_output_shapes
: W
Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 
Tensordot/concatConcatV2Tensordot/free:output:0Tensordot/axes:output:0Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:y
Tensordot/stackPackTensordot/Prod:output:0Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:y
Tensordot/transpose	TransposeinputsTensordot/concat:output:0*
T0*+
_output_shapes
:џџџџџџџџџ2y
Tensordot/ReshapeReshapeTensordot/transpose:y:0Tensordot/stack:output:0*
T0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ
Tensordot/MatMulMatMulTensordot/Reshape:output:0 Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ[
Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:Y
Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : Ї
Tensordot/concat_1ConcatV2Tensordot/GatherV2:output:0Tensordot/Const_2:output:0 Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:
	TensordotReshapeTensordot/MatMul:product:0Tensordot/concat_1:output:0*
T0*+
_output_shapes
:џџџџџџџџџ2r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0|
BiasAddBiasAddTensordot:output:0BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:џџџџџџџџџ2Z
SigmoidSigmoidBiasAdd:output:0*
T0*+
_output_shapes
:џџџџџџџџџ2^
IdentityIdentitySigmoid:y:0^NoOp*
T0*+
_output_shapes
:џџџџџџџџџ2z
NoOpNoOp^BiasAdd/ReadVariableOp^Tensordot/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:џџџџџџџџџ2y: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp24
Tensordot/ReadVariableOpTensordot/ReadVariableOp:S O
+
_output_shapes
:џџџџџџџџџ2y
 
_user_specified_nameinputs
ќ

E__inference_conv1d_82_layer_call_and_return_conditional_losses_799837

inputsA
+conv1d_expanddims_1_readvariableop_resource:ja-
biasadd_readvariableop_resource:a

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
:џџџџџџџџџ2
"Conv1D/ExpandDims_1/ReadVariableOpReadVariableOp+conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:ja*
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
:jaЌ
Conv1DConv2DConv1D/ExpandDims:output:0Conv1D/ExpandDims_1:output:0*
T0*/
_output_shapes
:џџџџџџџџџa*
paddingSAME*
strides
=
Conv1D/SqueezeSqueezeConv1D:output:0*
T0*+
_output_shapes
:џџџџџџџџџa*
squeeze_dims

§џџџџџџџџr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:a*
dtype0
BiasAddBiasAddConv1D/Squeeze:output:0BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:џџџџџџџџџaI
betaConst*
_output_shapes
: *
dtype0*
valueB
 *  ?a
mulMulbeta:output:0BiasAdd:output:0*
T0*+
_output_shapes
:џџџџџџџџџaQ
SigmoidSigmoidmul:z:0*
T0*+
_output_shapes
:џџџџџџџџџaa
mul_1MulBiasAdd:output:0Sigmoid:y:0*
T0*+
_output_shapes
:џџџџџџџџџaU
IdentityIdentity	mul_1:z:0*
T0*+
_output_shapes
:џџџџџџџџџaФ
	IdentityN	IdentityN	mul_1:z:0BiasAdd:output:0beta:output:0*
T
2*,
_gradient_op_typeCustomGradient-799828*D
_output_shapes2
0:џџџџџџџџџa:џџџџџџџџџa: g

Identity_1IdentityIdentityN:output:0^NoOp*
T0*+
_output_shapes
:џџџџџџџџџa
NoOpNoOp^BiasAdd/ReadVariableOp#^Conv1D/ExpandDims_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:џџџџџџџџџ2: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2H
"Conv1D/ExpandDims_1/ReadVariableOp"Conv1D/ExpandDims_1/ReadVariableOp:S O
+
_output_shapes
:џџџџџџџџџ2
 
_user_specified_nameinputs
Ђџ
э
__inference__traced_save_800421
file_prefix5
read_disablecopyonread_kernel_4:Fy-
read_1_disablecopyonread_bias_4:y3
!read_2_disablecopyonread_kernel_3:y-
read_3_disablecopyonread_bias_3:7
!read_4_disablecopyonread_kernel_2:ja-
read_5_disablecopyonread_bias_2:a7
!read_6_disablecopyonread_kernel_1:sak-
read_7_disablecopyonread_bias_1:k1
read_8_disablecopyonread_kernel:k+
read_9_disablecopyonread_bias:-
#read_10_disablecopyonread_iteration:	 1
'read_11_disablecopyonread_learning_rate: ?
)read_12_disablecopyonread_adam_m_kernel_4:Fy?
)read_13_disablecopyonread_adam_v_kernel_4:Fy5
'read_14_disablecopyonread_adam_m_bias_4:y5
'read_15_disablecopyonread_adam_v_bias_4:y;
)read_16_disablecopyonread_adam_m_kernel_3:y;
)read_17_disablecopyonread_adam_v_kernel_3:y5
'read_18_disablecopyonread_adam_m_bias_3:5
'read_19_disablecopyonread_adam_v_bias_3:?
)read_20_disablecopyonread_adam_m_kernel_2:ja?
)read_21_disablecopyonread_adam_v_kernel_2:ja5
'read_22_disablecopyonread_adam_m_bias_2:a5
'read_23_disablecopyonread_adam_v_bias_2:a?
)read_24_disablecopyonread_adam_m_kernel_1:sak?
)read_25_disablecopyonread_adam_v_kernel_1:sak5
'read_26_disablecopyonread_adam_m_bias_1:k5
'read_27_disablecopyonread_adam_v_bias_1:k9
'read_28_disablecopyonread_adam_m_kernel:k9
'read_29_disablecopyonread_adam_v_kernel:k3
%read_30_disablecopyonread_adam_m_bias:3
%read_31_disablecopyonread_adam_v_bias:+
!read_32_disablecopyonread_total_1: +
!read_33_disablecopyonread_count_1: )
read_34_disablecopyonread_total: )
read_35_disablecopyonread_count: 
savev2_const
identity_73ЂMergeV2CheckpointsЂRead/DisableCopyOnReadЂRead/ReadVariableOpЂRead_1/DisableCopyOnReadЂRead_1/ReadVariableOpЂRead_10/DisableCopyOnReadЂRead_10/ReadVariableOpЂRead_11/DisableCopyOnReadЂRead_11/ReadVariableOpЂRead_12/DisableCopyOnReadЂRead_12/ReadVariableOpЂRead_13/DisableCopyOnReadЂRead_13/ReadVariableOpЂRead_14/DisableCopyOnReadЂRead_14/ReadVariableOpЂRead_15/DisableCopyOnReadЂRead_15/ReadVariableOpЂRead_16/DisableCopyOnReadЂRead_16/ReadVariableOpЂRead_17/DisableCopyOnReadЂRead_17/ReadVariableOpЂRead_18/DisableCopyOnReadЂRead_18/ReadVariableOpЂRead_19/DisableCopyOnReadЂRead_19/ReadVariableOpЂRead_2/DisableCopyOnReadЂRead_2/ReadVariableOpЂRead_20/DisableCopyOnReadЂRead_20/ReadVariableOpЂRead_21/DisableCopyOnReadЂRead_21/ReadVariableOpЂRead_22/DisableCopyOnReadЂRead_22/ReadVariableOpЂRead_23/DisableCopyOnReadЂRead_23/ReadVariableOpЂRead_24/DisableCopyOnReadЂRead_24/ReadVariableOpЂRead_25/DisableCopyOnReadЂRead_25/ReadVariableOpЂRead_26/DisableCopyOnReadЂRead_26/ReadVariableOpЂRead_27/DisableCopyOnReadЂRead_27/ReadVariableOpЂRead_28/DisableCopyOnReadЂRead_28/ReadVariableOpЂRead_29/DisableCopyOnReadЂRead_29/ReadVariableOpЂRead_3/DisableCopyOnReadЂRead_3/ReadVariableOpЂRead_30/DisableCopyOnReadЂRead_30/ReadVariableOpЂRead_31/DisableCopyOnReadЂRead_31/ReadVariableOpЂRead_32/DisableCopyOnReadЂRead_32/ReadVariableOpЂRead_33/DisableCopyOnReadЂRead_33/ReadVariableOpЂRead_34/DisableCopyOnReadЂRead_34/ReadVariableOpЂRead_35/DisableCopyOnReadЂRead_35/ReadVariableOpЂRead_4/DisableCopyOnReadЂRead_4/ReadVariableOpЂRead_5/DisableCopyOnReadЂRead_5/ReadVariableOpЂRead_6/DisableCopyOnReadЂRead_6/ReadVariableOpЂRead_7/DisableCopyOnReadЂRead_7/ReadVariableOpЂRead_8/DisableCopyOnReadЂRead_8/ReadVariableOpЂRead_9/DisableCopyOnReadЂRead_9/ReadVariableOpw
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
Read/DisableCopyOnReadDisableCopyOnReadread_disablecopyonread_kernel_4"/device:CPU:0*
_output_shapes
 
Read/ReadVariableOpReadVariableOpread_disablecopyonread_kernel_4^Read/DisableCopyOnRead"/device:CPU:0*"
_output_shapes
:Fy*
dtype0m
IdentityIdentityRead/ReadVariableOp:value:0"/device:CPU:0*
T0*"
_output_shapes
:Fye

Identity_1IdentityIdentity:output:0"/device:CPU:0*
T0*"
_output_shapes
:Fys
Read_1/DisableCopyOnReadDisableCopyOnReadread_1_disablecopyonread_bias_4"/device:CPU:0*
_output_shapes
 
Read_1/ReadVariableOpReadVariableOpread_1_disablecopyonread_bias_4^Read_1/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:y*
dtype0i

Identity_2IdentityRead_1/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:y_

Identity_3IdentityIdentity_2:output:0"/device:CPU:0*
T0*
_output_shapes
:yu
Read_2/DisableCopyOnReadDisableCopyOnRead!read_2_disablecopyonread_kernel_3"/device:CPU:0*
_output_shapes
 Ё
Read_2/ReadVariableOpReadVariableOp!read_2_disablecopyonread_kernel_3^Read_2/DisableCopyOnRead"/device:CPU:0*
_output_shapes

:y*
dtype0m

Identity_4IdentityRead_2/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

:yc

Identity_5IdentityIdentity_4:output:0"/device:CPU:0*
T0*
_output_shapes

:ys
Read_3/DisableCopyOnReadDisableCopyOnReadread_3_disablecopyonread_bias_3"/device:CPU:0*
_output_shapes
 
Read_3/ReadVariableOpReadVariableOpread_3_disablecopyonread_bias_3^Read_3/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0i

Identity_6IdentityRead_3/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:_

Identity_7IdentityIdentity_6:output:0"/device:CPU:0*
T0*
_output_shapes
:u
Read_4/DisableCopyOnReadDisableCopyOnRead!read_4_disablecopyonread_kernel_2"/device:CPU:0*
_output_shapes
 Ѕ
Read_4/ReadVariableOpReadVariableOp!read_4_disablecopyonread_kernel_2^Read_4/DisableCopyOnRead"/device:CPU:0*"
_output_shapes
:ja*
dtype0q

Identity_8IdentityRead_4/ReadVariableOp:value:0"/device:CPU:0*
T0*"
_output_shapes
:jag

Identity_9IdentityIdentity_8:output:0"/device:CPU:0*
T0*"
_output_shapes
:jas
Read_5/DisableCopyOnReadDisableCopyOnReadread_5_disablecopyonread_bias_2"/device:CPU:0*
_output_shapes
 
Read_5/ReadVariableOpReadVariableOpread_5_disablecopyonread_bias_2^Read_5/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:a*
dtype0j
Identity_10IdentityRead_5/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:aa
Identity_11IdentityIdentity_10:output:0"/device:CPU:0*
T0*
_output_shapes
:au
Read_6/DisableCopyOnReadDisableCopyOnRead!read_6_disablecopyonread_kernel_1"/device:CPU:0*
_output_shapes
 Ѕ
Read_6/ReadVariableOpReadVariableOp!read_6_disablecopyonread_kernel_1^Read_6/DisableCopyOnRead"/device:CPU:0*"
_output_shapes
:sak*
dtype0r
Identity_12IdentityRead_6/ReadVariableOp:value:0"/device:CPU:0*
T0*"
_output_shapes
:saki
Identity_13IdentityIdentity_12:output:0"/device:CPU:0*
T0*"
_output_shapes
:saks
Read_7/DisableCopyOnReadDisableCopyOnReadread_7_disablecopyonread_bias_1"/device:CPU:0*
_output_shapes
 
Read_7/ReadVariableOpReadVariableOpread_7_disablecopyonread_bias_1^Read_7/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:k*
dtype0j
Identity_14IdentityRead_7/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:ka
Identity_15IdentityIdentity_14:output:0"/device:CPU:0*
T0*
_output_shapes
:ks
Read_8/DisableCopyOnReadDisableCopyOnReadread_8_disablecopyonread_kernel"/device:CPU:0*
_output_shapes
 
Read_8/ReadVariableOpReadVariableOpread_8_disablecopyonread_kernel^Read_8/DisableCopyOnRead"/device:CPU:0*
_output_shapes

:k*
dtype0n
Identity_16IdentityRead_8/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

:ke
Identity_17IdentityIdentity_16:output:0"/device:CPU:0*
T0*
_output_shapes

:kq
Read_9/DisableCopyOnReadDisableCopyOnReadread_9_disablecopyonread_bias"/device:CPU:0*
_output_shapes
 
Read_9/ReadVariableOpReadVariableOpread_9_disablecopyonread_bias^Read_9/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0j
Identity_18IdentityRead_9/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:a
Identity_19IdentityIdentity_18:output:0"/device:CPU:0*
T0*
_output_shapes
:x
Read_10/DisableCopyOnReadDisableCopyOnRead#read_10_disablecopyonread_iteration"/device:CPU:0*
_output_shapes
 
Read_10/ReadVariableOpReadVariableOp#read_10_disablecopyonread_iteration^Read_10/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0	g
Identity_20IdentityRead_10/ReadVariableOp:value:0"/device:CPU:0*
T0	*
_output_shapes
: ]
Identity_21IdentityIdentity_20:output:0"/device:CPU:0*
T0	*
_output_shapes
: |
Read_11/DisableCopyOnReadDisableCopyOnRead'read_11_disablecopyonread_learning_rate"/device:CPU:0*
_output_shapes
 Ё
Read_11/ReadVariableOpReadVariableOp'read_11_disablecopyonread_learning_rate^Read_11/DisableCopyOnRead"/device:CPU:0*
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
: ~
Read_12/DisableCopyOnReadDisableCopyOnRead)read_12_disablecopyonread_adam_m_kernel_4"/device:CPU:0*
_output_shapes
 Џ
Read_12/ReadVariableOpReadVariableOp)read_12_disablecopyonread_adam_m_kernel_4^Read_12/DisableCopyOnRead"/device:CPU:0*"
_output_shapes
:Fy*
dtype0s
Identity_24IdentityRead_12/ReadVariableOp:value:0"/device:CPU:0*
T0*"
_output_shapes
:Fyi
Identity_25IdentityIdentity_24:output:0"/device:CPU:0*
T0*"
_output_shapes
:Fy~
Read_13/DisableCopyOnReadDisableCopyOnRead)read_13_disablecopyonread_adam_v_kernel_4"/device:CPU:0*
_output_shapes
 Џ
Read_13/ReadVariableOpReadVariableOp)read_13_disablecopyonread_adam_v_kernel_4^Read_13/DisableCopyOnRead"/device:CPU:0*"
_output_shapes
:Fy*
dtype0s
Identity_26IdentityRead_13/ReadVariableOp:value:0"/device:CPU:0*
T0*"
_output_shapes
:Fyi
Identity_27IdentityIdentity_26:output:0"/device:CPU:0*
T0*"
_output_shapes
:Fy|
Read_14/DisableCopyOnReadDisableCopyOnRead'read_14_disablecopyonread_adam_m_bias_4"/device:CPU:0*
_output_shapes
 Ѕ
Read_14/ReadVariableOpReadVariableOp'read_14_disablecopyonread_adam_m_bias_4^Read_14/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:y*
dtype0k
Identity_28IdentityRead_14/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:ya
Identity_29IdentityIdentity_28:output:0"/device:CPU:0*
T0*
_output_shapes
:y|
Read_15/DisableCopyOnReadDisableCopyOnRead'read_15_disablecopyonread_adam_v_bias_4"/device:CPU:0*
_output_shapes
 Ѕ
Read_15/ReadVariableOpReadVariableOp'read_15_disablecopyonread_adam_v_bias_4^Read_15/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:y*
dtype0k
Identity_30IdentityRead_15/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:ya
Identity_31IdentityIdentity_30:output:0"/device:CPU:0*
T0*
_output_shapes
:y~
Read_16/DisableCopyOnReadDisableCopyOnRead)read_16_disablecopyonread_adam_m_kernel_3"/device:CPU:0*
_output_shapes
 Ћ
Read_16/ReadVariableOpReadVariableOp)read_16_disablecopyonread_adam_m_kernel_3^Read_16/DisableCopyOnRead"/device:CPU:0*
_output_shapes

:y*
dtype0o
Identity_32IdentityRead_16/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

:ye
Identity_33IdentityIdentity_32:output:0"/device:CPU:0*
T0*
_output_shapes

:y~
Read_17/DisableCopyOnReadDisableCopyOnRead)read_17_disablecopyonread_adam_v_kernel_3"/device:CPU:0*
_output_shapes
 Ћ
Read_17/ReadVariableOpReadVariableOp)read_17_disablecopyonread_adam_v_kernel_3^Read_17/DisableCopyOnRead"/device:CPU:0*
_output_shapes

:y*
dtype0o
Identity_34IdentityRead_17/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

:ye
Identity_35IdentityIdentity_34:output:0"/device:CPU:0*
T0*
_output_shapes

:y|
Read_18/DisableCopyOnReadDisableCopyOnRead'read_18_disablecopyonread_adam_m_bias_3"/device:CPU:0*
_output_shapes
 Ѕ
Read_18/ReadVariableOpReadVariableOp'read_18_disablecopyonread_adam_m_bias_3^Read_18/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0k
Identity_36IdentityRead_18/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:a
Identity_37IdentityIdentity_36:output:0"/device:CPU:0*
T0*
_output_shapes
:|
Read_19/DisableCopyOnReadDisableCopyOnRead'read_19_disablecopyonread_adam_v_bias_3"/device:CPU:0*
_output_shapes
 Ѕ
Read_19/ReadVariableOpReadVariableOp'read_19_disablecopyonread_adam_v_bias_3^Read_19/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0k
Identity_38IdentityRead_19/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:a
Identity_39IdentityIdentity_38:output:0"/device:CPU:0*
T0*
_output_shapes
:~
Read_20/DisableCopyOnReadDisableCopyOnRead)read_20_disablecopyonread_adam_m_kernel_2"/device:CPU:0*
_output_shapes
 Џ
Read_20/ReadVariableOpReadVariableOp)read_20_disablecopyonread_adam_m_kernel_2^Read_20/DisableCopyOnRead"/device:CPU:0*"
_output_shapes
:ja*
dtype0s
Identity_40IdentityRead_20/ReadVariableOp:value:0"/device:CPU:0*
T0*"
_output_shapes
:jai
Identity_41IdentityIdentity_40:output:0"/device:CPU:0*
T0*"
_output_shapes
:ja~
Read_21/DisableCopyOnReadDisableCopyOnRead)read_21_disablecopyonread_adam_v_kernel_2"/device:CPU:0*
_output_shapes
 Џ
Read_21/ReadVariableOpReadVariableOp)read_21_disablecopyonread_adam_v_kernel_2^Read_21/DisableCopyOnRead"/device:CPU:0*"
_output_shapes
:ja*
dtype0s
Identity_42IdentityRead_21/ReadVariableOp:value:0"/device:CPU:0*
T0*"
_output_shapes
:jai
Identity_43IdentityIdentity_42:output:0"/device:CPU:0*
T0*"
_output_shapes
:ja|
Read_22/DisableCopyOnReadDisableCopyOnRead'read_22_disablecopyonread_adam_m_bias_2"/device:CPU:0*
_output_shapes
 Ѕ
Read_22/ReadVariableOpReadVariableOp'read_22_disablecopyonread_adam_m_bias_2^Read_22/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:a*
dtype0k
Identity_44IdentityRead_22/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:aa
Identity_45IdentityIdentity_44:output:0"/device:CPU:0*
T0*
_output_shapes
:a|
Read_23/DisableCopyOnReadDisableCopyOnRead'read_23_disablecopyonread_adam_v_bias_2"/device:CPU:0*
_output_shapes
 Ѕ
Read_23/ReadVariableOpReadVariableOp'read_23_disablecopyonread_adam_v_bias_2^Read_23/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:a*
dtype0k
Identity_46IdentityRead_23/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:aa
Identity_47IdentityIdentity_46:output:0"/device:CPU:0*
T0*
_output_shapes
:a~
Read_24/DisableCopyOnReadDisableCopyOnRead)read_24_disablecopyonread_adam_m_kernel_1"/device:CPU:0*
_output_shapes
 Џ
Read_24/ReadVariableOpReadVariableOp)read_24_disablecopyonread_adam_m_kernel_1^Read_24/DisableCopyOnRead"/device:CPU:0*"
_output_shapes
:sak*
dtype0s
Identity_48IdentityRead_24/ReadVariableOp:value:0"/device:CPU:0*
T0*"
_output_shapes
:saki
Identity_49IdentityIdentity_48:output:0"/device:CPU:0*
T0*"
_output_shapes
:sak~
Read_25/DisableCopyOnReadDisableCopyOnRead)read_25_disablecopyonread_adam_v_kernel_1"/device:CPU:0*
_output_shapes
 Џ
Read_25/ReadVariableOpReadVariableOp)read_25_disablecopyonread_adam_v_kernel_1^Read_25/DisableCopyOnRead"/device:CPU:0*"
_output_shapes
:sak*
dtype0s
Identity_50IdentityRead_25/ReadVariableOp:value:0"/device:CPU:0*
T0*"
_output_shapes
:saki
Identity_51IdentityIdentity_50:output:0"/device:CPU:0*
T0*"
_output_shapes
:sak|
Read_26/DisableCopyOnReadDisableCopyOnRead'read_26_disablecopyonread_adam_m_bias_1"/device:CPU:0*
_output_shapes
 Ѕ
Read_26/ReadVariableOpReadVariableOp'read_26_disablecopyonread_adam_m_bias_1^Read_26/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:k*
dtype0k
Identity_52IdentityRead_26/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:ka
Identity_53IdentityIdentity_52:output:0"/device:CPU:0*
T0*
_output_shapes
:k|
Read_27/DisableCopyOnReadDisableCopyOnRead'read_27_disablecopyonread_adam_v_bias_1"/device:CPU:0*
_output_shapes
 Ѕ
Read_27/ReadVariableOpReadVariableOp'read_27_disablecopyonread_adam_v_bias_1^Read_27/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:k*
dtype0k
Identity_54IdentityRead_27/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:ka
Identity_55IdentityIdentity_54:output:0"/device:CPU:0*
T0*
_output_shapes
:k|
Read_28/DisableCopyOnReadDisableCopyOnRead'read_28_disablecopyonread_adam_m_kernel"/device:CPU:0*
_output_shapes
 Љ
Read_28/ReadVariableOpReadVariableOp'read_28_disablecopyonread_adam_m_kernel^Read_28/DisableCopyOnRead"/device:CPU:0*
_output_shapes

:k*
dtype0o
Identity_56IdentityRead_28/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

:ke
Identity_57IdentityIdentity_56:output:0"/device:CPU:0*
T0*
_output_shapes

:k|
Read_29/DisableCopyOnReadDisableCopyOnRead'read_29_disablecopyonread_adam_v_kernel"/device:CPU:0*
_output_shapes
 Љ
Read_29/ReadVariableOpReadVariableOp'read_29_disablecopyonread_adam_v_kernel^Read_29/DisableCopyOnRead"/device:CPU:0*
_output_shapes

:k*
dtype0o
Identity_58IdentityRead_29/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

:ke
Identity_59IdentityIdentity_58:output:0"/device:CPU:0*
T0*
_output_shapes

:kz
Read_30/DisableCopyOnReadDisableCopyOnRead%read_30_disablecopyonread_adam_m_bias"/device:CPU:0*
_output_shapes
 Ѓ
Read_30/ReadVariableOpReadVariableOp%read_30_disablecopyonread_adam_m_bias^Read_30/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0k
Identity_60IdentityRead_30/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:a
Identity_61IdentityIdentity_60:output:0"/device:CPU:0*
T0*
_output_shapes
:z
Read_31/DisableCopyOnReadDisableCopyOnRead%read_31_disablecopyonread_adam_v_bias"/device:CPU:0*
_output_shapes
 Ѓ
Read_31/ReadVariableOpReadVariableOp%read_31_disablecopyonread_adam_v_bias^Read_31/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0k
Identity_62IdentityRead_31/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:a
Identity_63IdentityIdentity_62:output:0"/device:CPU:0*
T0*
_output_shapes
:v
Read_32/DisableCopyOnReadDisableCopyOnRead!read_32_disablecopyonread_total_1"/device:CPU:0*
_output_shapes
 
Read_32/ReadVariableOpReadVariableOp!read_32_disablecopyonread_total_1^Read_32/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0g
Identity_64IdentityRead_32/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: ]
Identity_65IdentityIdentity_64:output:0"/device:CPU:0*
T0*
_output_shapes
: v
Read_33/DisableCopyOnReadDisableCopyOnRead!read_33_disablecopyonread_count_1"/device:CPU:0*
_output_shapes
 
Read_33/ReadVariableOpReadVariableOp!read_33_disablecopyonread_count_1^Read_33/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0g
Identity_66IdentityRead_33/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: ]
Identity_67IdentityIdentity_66:output:0"/device:CPU:0*
T0*
_output_shapes
: t
Read_34/DisableCopyOnReadDisableCopyOnReadread_34_disablecopyonread_total"/device:CPU:0*
_output_shapes
 
Read_34/ReadVariableOpReadVariableOpread_34_disablecopyonread_total^Read_34/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0g
Identity_68IdentityRead_34/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: ]
Identity_69IdentityIdentity_68:output:0"/device:CPU:0*
T0*
_output_shapes
: t
Read_35/DisableCopyOnReadDisableCopyOnReadread_35_disablecopyonread_count"/device:CPU:0*
_output_shapes
 
Read_35/ReadVariableOpReadVariableOpread_35_disablecopyonread_count^Read_35/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0g
Identity_70IdentityRead_35/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: ]
Identity_71IdentityIdentity_70:output:0"/device:CPU:0*
T0*
_output_shapes
: і
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:%*
dtype0*
valueB%B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB0optimizer/_iterations/.ATTRIBUTES/VARIABLE_VALUEB3optimizer/_learning_rate/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/1/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/2/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/3/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/4/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/5/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/6/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/7/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/8/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/9/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/10/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/11/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/12/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/13/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/14/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/15/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/16/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/17/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/18/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/19/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/20/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPHЗ
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:%*
dtype0*]
valueTBR%B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0Identity_1:output:0Identity_3:output:0Identity_5:output:0Identity_7:output:0Identity_9:output:0Identity_11:output:0Identity_13:output:0Identity_15:output:0Identity_17:output:0Identity_19:output:0Identity_21:output:0Identity_23:output:0Identity_25:output:0Identity_27:output:0Identity_29:output:0Identity_31:output:0Identity_33:output:0Identity_35:output:0Identity_37:output:0Identity_39:output:0Identity_41:output:0Identity_43:output:0Identity_45:output:0Identity_47:output:0Identity_49:output:0Identity_51:output:0Identity_53:output:0Identity_55:output:0Identity_57:output:0Identity_59:output:0Identity_61:output:0Identity_63:output:0Identity_65:output:0Identity_67:output:0Identity_69:output:0Identity_71:output:0savev2_const"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *3
dtypes)
'2%	
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:Г
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 i
Identity_72Identityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: U
Identity_73IdentityIdentity_72:output:0^NoOp*
T0*
_output_shapes
: З
NoOpNoOp^MergeV2Checkpoints^Read/DisableCopyOnRead^Read/ReadVariableOp^Read_1/DisableCopyOnRead^Read_1/ReadVariableOp^Read_10/DisableCopyOnRead^Read_10/ReadVariableOp^Read_11/DisableCopyOnRead^Read_11/ReadVariableOp^Read_12/DisableCopyOnRead^Read_12/ReadVariableOp^Read_13/DisableCopyOnRead^Read_13/ReadVariableOp^Read_14/DisableCopyOnRead^Read_14/ReadVariableOp^Read_15/DisableCopyOnRead^Read_15/ReadVariableOp^Read_16/DisableCopyOnRead^Read_16/ReadVariableOp^Read_17/DisableCopyOnRead^Read_17/ReadVariableOp^Read_18/DisableCopyOnRead^Read_18/ReadVariableOp^Read_19/DisableCopyOnRead^Read_19/ReadVariableOp^Read_2/DisableCopyOnRead^Read_2/ReadVariableOp^Read_20/DisableCopyOnRead^Read_20/ReadVariableOp^Read_21/DisableCopyOnRead^Read_21/ReadVariableOp^Read_22/DisableCopyOnRead^Read_22/ReadVariableOp^Read_23/DisableCopyOnRead^Read_23/ReadVariableOp^Read_24/DisableCopyOnRead^Read_24/ReadVariableOp^Read_25/DisableCopyOnRead^Read_25/ReadVariableOp^Read_26/DisableCopyOnRead^Read_26/ReadVariableOp^Read_27/DisableCopyOnRead^Read_27/ReadVariableOp^Read_28/DisableCopyOnRead^Read_28/ReadVariableOp^Read_29/DisableCopyOnRead^Read_29/ReadVariableOp^Read_3/DisableCopyOnRead^Read_3/ReadVariableOp^Read_30/DisableCopyOnRead^Read_30/ReadVariableOp^Read_31/DisableCopyOnRead^Read_31/ReadVariableOp^Read_32/DisableCopyOnRead^Read_32/ReadVariableOp^Read_33/DisableCopyOnRead^Read_33/ReadVariableOp^Read_34/DisableCopyOnRead^Read_34/ReadVariableOp^Read_35/DisableCopyOnRead^Read_35/ReadVariableOp^Read_4/DisableCopyOnRead^Read_4/ReadVariableOp^Read_5/DisableCopyOnRead^Read_5/ReadVariableOp^Read_6/DisableCopyOnRead^Read_6/ReadVariableOp^Read_7/DisableCopyOnRead^Read_7/ReadVariableOp^Read_8/DisableCopyOnRead^Read_8/ReadVariableOp^Read_9/DisableCopyOnRead^Read_9/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "#
identity_73Identity_73:output:0*_
_input_shapesN
L: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2(
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
Read_17/ReadVariableOpRead_17/ReadVariableOp26
Read_18/DisableCopyOnReadRead_18/DisableCopyOnRead20
Read_18/ReadVariableOpRead_18/ReadVariableOp26
Read_19/DisableCopyOnReadRead_19/DisableCopyOnRead20
Read_19/ReadVariableOpRead_19/ReadVariableOp24
Read_2/DisableCopyOnReadRead_2/DisableCopyOnRead2.
Read_2/ReadVariableOpRead_2/ReadVariableOp26
Read_20/DisableCopyOnReadRead_20/DisableCopyOnRead20
Read_20/ReadVariableOpRead_20/ReadVariableOp26
Read_21/DisableCopyOnReadRead_21/DisableCopyOnRead20
Read_21/ReadVariableOpRead_21/ReadVariableOp26
Read_22/DisableCopyOnReadRead_22/DisableCopyOnRead20
Read_22/ReadVariableOpRead_22/ReadVariableOp26
Read_23/DisableCopyOnReadRead_23/DisableCopyOnRead20
Read_23/ReadVariableOpRead_23/ReadVariableOp26
Read_24/DisableCopyOnReadRead_24/DisableCopyOnRead20
Read_24/ReadVariableOpRead_24/ReadVariableOp26
Read_25/DisableCopyOnReadRead_25/DisableCopyOnRead20
Read_25/ReadVariableOpRead_25/ReadVariableOp26
Read_26/DisableCopyOnReadRead_26/DisableCopyOnRead20
Read_26/ReadVariableOpRead_26/ReadVariableOp26
Read_27/DisableCopyOnReadRead_27/DisableCopyOnRead20
Read_27/ReadVariableOpRead_27/ReadVariableOp26
Read_28/DisableCopyOnReadRead_28/DisableCopyOnRead20
Read_28/ReadVariableOpRead_28/ReadVariableOp26
Read_29/DisableCopyOnReadRead_29/DisableCopyOnRead20
Read_29/ReadVariableOpRead_29/ReadVariableOp24
Read_3/DisableCopyOnReadRead_3/DisableCopyOnRead2.
Read_3/ReadVariableOpRead_3/ReadVariableOp26
Read_30/DisableCopyOnReadRead_30/DisableCopyOnRead20
Read_30/ReadVariableOpRead_30/ReadVariableOp26
Read_31/DisableCopyOnReadRead_31/DisableCopyOnRead20
Read_31/ReadVariableOpRead_31/ReadVariableOp26
Read_32/DisableCopyOnReadRead_32/DisableCopyOnRead20
Read_32/ReadVariableOpRead_32/ReadVariableOp26
Read_33/DisableCopyOnReadRead_33/DisableCopyOnRead20
Read_33/ReadVariableOpRead_33/ReadVariableOp26
Read_34/DisableCopyOnReadRead_34/DisableCopyOnRead20
Read_34/ReadVariableOpRead_34/ReadVariableOp26
Read_35/DisableCopyOnReadRead_35/DisableCopyOnRead20
Read_35/ReadVariableOpRead_35/ReadVariableOp24
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
Read_9/ReadVariableOpRead_9/ReadVariableOp:%

_output_shapes
: :C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
Щ
C
__inference_crop_samples_1151
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
 
Ф
#__inference_internal_grad_fn_800229
result_grads_0
result_grads_1
result_grads_2
mul_model_76_conv1d_82_beta"
mul_model_76_conv1d_82_biasadd
identity

identity_1
mulMulmul_model_76_conv1d_82_betamul_model_76_conv1d_82_biasadd^result_grads_0*
T0*+
_output_shapes
:џџџџџџџџџaQ
SigmoidSigmoidmul:z:0*
T0*+
_output_shapes
:џџџџџџџџџa
mul_1Mulmul_model_76_conv1d_82_betamul_model_76_conv1d_82_biasadd*
T0*+
_output_shapes
:џџџџџџџџџaJ
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?]
subSubsub/x:output:0Sigmoid:y:0*
T0*+
_output_shapes
:џџџџџџџџџaV
mul_2Mul	mul_1:z:0sub:z:0*
T0*+
_output_shapes
:џџџџџџџџџaJ
add/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?]
addAddV2add/x:output:0	mul_2:z:0*
T0*+
_output_shapes
:џџџџџџџџџaX
mul_3MulSigmoid:y:0add:z:0*
T0*+
_output_shapes
:џџџџџџџџџaf
SquareSquaremul_model_76_conv1d_82_biasadd*
T0*+
_output_shapes
:џџџџџџџџџa^
mul_4Mulresult_grads_0
Square:y:0*
T0*+
_output_shapes
:џџџџџџџџџaZ
mul_5Mul	mul_4:z:0Sigmoid:y:0*
T0*+
_output_shapes
:џџџџџџџџџaL
sub_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?a
sub_1Subsub_1/x:output:0Sigmoid:y:0*
T0*+
_output_shapes
:џџџџџџџџџaX
mul_6Mul	mul_5:z:0	sub_1:z:0*
T0*+
_output_shapes
:џџџџџџџџџaZ
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
:џџџџџџџџџaU
IdentityIdentity	mul_7:z:0*
T0*+
_output_shapes
:џџџџџџџџџaE

Identity_1IdentitySum:output:0*
T0*
_output_shapes
: "!

identity_1Identity_1:output:0"
identityIdentity:output:0*\
_input_shapesK
I:џџџџџџџџџa:џџџџџџџџџa: : :џџџџџџџџџa:1-
+
_output_shapes
:џџџџџџџџџa:
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
:џџџџџџџџџa
(
_user_specified_nameresult_grads_1: 
&
 _has_manual_control_dependencies(
+
_output_shapes
:џџџџџџџџџa
(
_user_specified_nameresult_grads_0
}


!__inference__wrapped_model_798986
	offsource
onsourceT
>model_76_conv1d_81_conv1d_expanddims_1_readvariableop_resource:Fy@
2model_76_conv1d_81_biasadd_readvariableop_resource:yE
3model_76_dense_89_tensordot_readvariableop_resource:y?
1model_76_dense_89_biasadd_readvariableop_resource:T
>model_76_conv1d_82_conv1d_expanddims_1_readvariableop_resource:ja@
2model_76_conv1d_82_biasadd_readvariableop_resource:aT
>model_76_conv1d_83_conv1d_expanddims_1_readvariableop_resource:sak@
2model_76_conv1d_83_biasadd_readvariableop_resource:kI
7model_76_injection_masks_matmul_readvariableop_resource:kF
8model_76_injection_masks_biasadd_readvariableop_resource:
identityЂ/model_76/INJECTION_MASKS/BiasAdd/ReadVariableOpЂ.model_76/INJECTION_MASKS/MatMul/ReadVariableOpЂ)model_76/conv1d_81/BiasAdd/ReadVariableOpЂ5model_76/conv1d_81/Conv1D/ExpandDims_1/ReadVariableOpЂ)model_76/conv1d_82/BiasAdd/ReadVariableOpЂ5model_76/conv1d_82/Conv1D/ExpandDims_1/ReadVariableOpЂ)model_76/conv1d_83/BiasAdd/ReadVariableOpЂ5model_76/conv1d_83/Conv1D/ExpandDims_1/ReadVariableOpЂ(model_76/dense_89/BiasAdd/ReadVariableOpЂ*model_76/dense_89/Tensordot/ReadVariableOpЪ
.model_76/whiten_passthrough_38/PartitionedCallPartitionedCall	offsource*
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
)__inference_restored_function_body_285667э
#model_76/reshape_76/PartitionedCallPartitionedCall7model_76/whiten_passthrough_38/PartitionedCall:output:0*
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
)__inference_restored_function_body_285673s
(model_76/conv1d_81/Conv1D/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
§џџџџџџџџЮ
$model_76/conv1d_81/Conv1D/ExpandDims
ExpandDims,model_76/reshape_76/PartitionedCall:output:01model_76/conv1d_81/Conv1D/ExpandDims/dim:output:0*
T0*0
_output_shapes
:џџџџџџџџџИ
5model_76/conv1d_81/Conv1D/ExpandDims_1/ReadVariableOpReadVariableOp>model_76_conv1d_81_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:Fy*
dtype0l
*model_76/conv1d_81/Conv1D/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : й
&model_76/conv1d_81/Conv1D/ExpandDims_1
ExpandDims=model_76/conv1d_81/Conv1D/ExpandDims_1/ReadVariableOp:value:03model_76/conv1d_81/Conv1D/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:Fyх
model_76/conv1d_81/Conv1DConv2D-model_76/conv1d_81/Conv1D/ExpandDims:output:0/model_76/conv1d_81/Conv1D/ExpandDims_1:output:0*
T0*/
_output_shapes
:џџџџџџџџџ2y*
paddingSAME*
strides
)І
!model_76/conv1d_81/Conv1D/SqueezeSqueeze"model_76/conv1d_81/Conv1D:output:0*
T0*+
_output_shapes
:џџџџџџџџџ2y*
squeeze_dims

§џџџџџџџџ
)model_76/conv1d_81/BiasAdd/ReadVariableOpReadVariableOp2model_76_conv1d_81_biasadd_readvariableop_resource*
_output_shapes
:y*
dtype0К
model_76/conv1d_81/BiasAddBiasAdd*model_76/conv1d_81/Conv1D/Squeeze:output:01model_76/conv1d_81/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:џџџџџџџџџ2y
model_76/conv1d_81/SigmoidSigmoid#model_76/conv1d_81/BiasAdd:output:0*
T0*+
_output_shapes
:џџџџџџџџџ2y~
model_76/dropout_83/IdentityIdentitymodel_76/conv1d_81/Sigmoid:y:0*
T0*+
_output_shapes
:џџџџџџџџџ2y
*model_76/dense_89/Tensordot/ReadVariableOpReadVariableOp3model_76_dense_89_tensordot_readvariableop_resource*
_output_shapes

:y*
dtype0j
 model_76/dense_89/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:q
 model_76/dense_89/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       
!model_76/dense_89/Tensordot/ShapeShape%model_76/dropout_83/Identity:output:0*
T0*
_output_shapes
::эЯk
)model_76/dense_89/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 
$model_76/dense_89/Tensordot/GatherV2GatherV2*model_76/dense_89/Tensordot/Shape:output:0)model_76/dense_89/Tensordot/free:output:02model_76/dense_89/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:m
+model_76/dense_89/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 
&model_76/dense_89/Tensordot/GatherV2_1GatherV2*model_76/dense_89/Tensordot/Shape:output:0)model_76/dense_89/Tensordot/axes:output:04model_76/dense_89/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:k
!model_76/dense_89/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: Є
 model_76/dense_89/Tensordot/ProdProd-model_76/dense_89/Tensordot/GatherV2:output:0*model_76/dense_89/Tensordot/Const:output:0*
T0*
_output_shapes
: m
#model_76/dense_89/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: Њ
"model_76/dense_89/Tensordot/Prod_1Prod/model_76/dense_89/Tensordot/GatherV2_1:output:0,model_76/dense_89/Tensordot/Const_1:output:0*
T0*
_output_shapes
: i
'model_76/dense_89/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ф
"model_76/dense_89/Tensordot/concatConcatV2)model_76/dense_89/Tensordot/free:output:0)model_76/dense_89/Tensordot/axes:output:00model_76/dense_89/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:Џ
!model_76/dense_89/Tensordot/stackPack)model_76/dense_89/Tensordot/Prod:output:0+model_76/dense_89/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:М
%model_76/dense_89/Tensordot/transpose	Transpose%model_76/dropout_83/Identity:output:0+model_76/dense_89/Tensordot/concat:output:0*
T0*+
_output_shapes
:џџџџџџџџџ2yР
#model_76/dense_89/Tensordot/ReshapeReshape)model_76/dense_89/Tensordot/transpose:y:0*model_76/dense_89/Tensordot/stack:output:0*
T0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџР
"model_76/dense_89/Tensordot/MatMulMatMul,model_76/dense_89/Tensordot/Reshape:output:02model_76/dense_89/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџm
#model_76/dense_89/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:k
)model_76/dense_89/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : я
$model_76/dense_89/Tensordot/concat_1ConcatV2-model_76/dense_89/Tensordot/GatherV2:output:0,model_76/dense_89/Tensordot/Const_2:output:02model_76/dense_89/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:Й
model_76/dense_89/TensordotReshape,model_76/dense_89/Tensordot/MatMul:product:0-model_76/dense_89/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:џџџџџџџџџ2
(model_76/dense_89/BiasAdd/ReadVariableOpReadVariableOp1model_76_dense_89_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0В
model_76/dense_89/BiasAddBiasAdd$model_76/dense_89/Tensordot:output:00model_76/dense_89/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:џџџџџџџџџ2~
model_76/dense_89/SigmoidSigmoid"model_76/dense_89/BiasAdd:output:0*
T0*+
_output_shapes
:џџџџџџџџџ2s
(model_76/conv1d_82/Conv1D/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
§џџџџџџџџО
$model_76/conv1d_82/Conv1D/ExpandDims
ExpandDimsmodel_76/dense_89/Sigmoid:y:01model_76/conv1d_82/Conv1D/ExpandDims/dim:output:0*
T0*/
_output_shapes
:џџџџџџџџџ2И
5model_76/conv1d_82/Conv1D/ExpandDims_1/ReadVariableOpReadVariableOp>model_76_conv1d_82_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:ja*
dtype0l
*model_76/conv1d_82/Conv1D/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : й
&model_76/conv1d_82/Conv1D/ExpandDims_1
ExpandDims=model_76/conv1d_82/Conv1D/ExpandDims_1/ReadVariableOp:value:03model_76/conv1d_82/Conv1D/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:jaх
model_76/conv1d_82/Conv1DConv2D-model_76/conv1d_82/Conv1D/ExpandDims:output:0/model_76/conv1d_82/Conv1D/ExpandDims_1:output:0*
T0*/
_output_shapes
:џџџџџџџџџa*
paddingSAME*
strides
=І
!model_76/conv1d_82/Conv1D/SqueezeSqueeze"model_76/conv1d_82/Conv1D:output:0*
T0*+
_output_shapes
:џџџџџџџџџa*
squeeze_dims

§џџџџџџџџ
)model_76/conv1d_82/BiasAdd/ReadVariableOpReadVariableOp2model_76_conv1d_82_biasadd_readvariableop_resource*
_output_shapes
:a*
dtype0К
model_76/conv1d_82/BiasAddBiasAdd*model_76/conv1d_82/Conv1D/Squeeze:output:01model_76/conv1d_82/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:џџџџџџџџџa\
model_76/conv1d_82/betaConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
model_76/conv1d_82/mulMul model_76/conv1d_82/beta:output:0#model_76/conv1d_82/BiasAdd:output:0*
T0*+
_output_shapes
:џџџџџџџџџaw
model_76/conv1d_82/SigmoidSigmoidmodel_76/conv1d_82/mul:z:0*
T0*+
_output_shapes
:џџџџџџџџџa
model_76/conv1d_82/mul_1Mul#model_76/conv1d_82/BiasAdd:output:0model_76/conv1d_82/Sigmoid:y:0*
T0*+
_output_shapes
:џџџџџџџџџa{
model_76/conv1d_82/IdentityIdentitymodel_76/conv1d_82/mul_1:z:0*
T0*+
_output_shapes
:џџџџџџџџџa
model_76/conv1d_82/IdentityN	IdentityNmodel_76/conv1d_82/mul_1:z:0#model_76/conv1d_82/BiasAdd:output:0 model_76/conv1d_82/beta:output:0*
T
2*,
_gradient_op_typeCustomGradient-798948*D
_output_shapes2
0:џџџџџџџџџa:џџџџџџџџџa: s
(model_76/conv1d_83/Conv1D/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
§џџџџџџџџЦ
$model_76/conv1d_83/Conv1D/ExpandDims
ExpandDims%model_76/conv1d_82/IdentityN:output:01model_76/conv1d_83/Conv1D/ExpandDims/dim:output:0*
T0*/
_output_shapes
:џџџџџџџџџaИ
5model_76/conv1d_83/Conv1D/ExpandDims_1/ReadVariableOpReadVariableOp>model_76_conv1d_83_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:sak*
dtype0l
*model_76/conv1d_83/Conv1D/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : й
&model_76/conv1d_83/Conv1D/ExpandDims_1
ExpandDims=model_76/conv1d_83/Conv1D/ExpandDims_1/ReadVariableOp:value:03model_76/conv1d_83/Conv1D/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:sakх
model_76/conv1d_83/Conv1DConv2D-model_76/conv1d_83/Conv1D/ExpandDims:output:0/model_76/conv1d_83/Conv1D/ExpandDims_1:output:0*
T0*/
_output_shapes
:џџџџџџџџџk*
paddingSAME*
strides
CІ
!model_76/conv1d_83/Conv1D/SqueezeSqueeze"model_76/conv1d_83/Conv1D:output:0*
T0*+
_output_shapes
:џџџџџџџџџk*
squeeze_dims

§џџџџџџџџ
)model_76/conv1d_83/BiasAdd/ReadVariableOpReadVariableOp2model_76_conv1d_83_biasadd_readvariableop_resource*
_output_shapes
:k*
dtype0К
model_76/conv1d_83/BiasAddBiasAdd*model_76/conv1d_83/Conv1D/Squeeze:output:01model_76/conv1d_83/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:џџџџџџџџџk\
model_76/conv1d_83/betaConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
model_76/conv1d_83/mulMul model_76/conv1d_83/beta:output:0#model_76/conv1d_83/BiasAdd:output:0*
T0*+
_output_shapes
:џџџџџџџџџkw
model_76/conv1d_83/SigmoidSigmoidmodel_76/conv1d_83/mul:z:0*
T0*+
_output_shapes
:џџџџџџџџџk
model_76/conv1d_83/mul_1Mul#model_76/conv1d_83/BiasAdd:output:0model_76/conv1d_83/Sigmoid:y:0*
T0*+
_output_shapes
:џџџџџџџџџk{
model_76/conv1d_83/IdentityIdentitymodel_76/conv1d_83/mul_1:z:0*
T0*+
_output_shapes
:џџџџџџџџџk
model_76/conv1d_83/IdentityN	IdentityNmodel_76/conv1d_83/mul_1:z:0#model_76/conv1d_83/BiasAdd:output:0 model_76/conv1d_83/beta:output:0*
T
2*,
_gradient_op_typeCustomGradient-798968*D
_output_shapes2
0:џџџџџџџџџk:џџџџџџџџџk: j
model_76/flatten_76/ConstConst*
_output_shapes
:*
dtype0*
valueB"џџџџk   Ѓ
model_76/flatten_76/ReshapeReshape%model_76/conv1d_83/IdentityN:output:0"model_76/flatten_76/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџkІ
.model_76/INJECTION_MASKS/MatMul/ReadVariableOpReadVariableOp7model_76_injection_masks_matmul_readvariableop_resource*
_output_shapes

:k*
dtype0Й
model_76/INJECTION_MASKS/MatMulMatMul$model_76/flatten_76/Reshape:output:06model_76/INJECTION_MASKS/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџЄ
/model_76/INJECTION_MASKS/BiasAdd/ReadVariableOpReadVariableOp8model_76_injection_masks_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0С
 model_76/INJECTION_MASKS/BiasAddBiasAdd)model_76/INJECTION_MASKS/MatMul:product:07model_76/INJECTION_MASKS/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ
 model_76/INJECTION_MASKS/SigmoidSigmoid)model_76/INJECTION_MASKS/BiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџs
IdentityIdentity$model_76/INJECTION_MASKS/Sigmoid:y:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ­
NoOpNoOp0^model_76/INJECTION_MASKS/BiasAdd/ReadVariableOp/^model_76/INJECTION_MASKS/MatMul/ReadVariableOp*^model_76/conv1d_81/BiasAdd/ReadVariableOp6^model_76/conv1d_81/Conv1D/ExpandDims_1/ReadVariableOp*^model_76/conv1d_82/BiasAdd/ReadVariableOp6^model_76/conv1d_82/Conv1D/ExpandDims_1/ReadVariableOp*^model_76/conv1d_83/BiasAdd/ReadVariableOp6^model_76/conv1d_83/Conv1D/ExpandDims_1/ReadVariableOp)^model_76/dense_89/BiasAdd/ReadVariableOp+^model_76/dense_89/Tensordot/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*X
_input_shapesG
E:џџџџџџџџџ:џџџџџџџџџ : : : : : : : : : : 2b
/model_76/INJECTION_MASKS/BiasAdd/ReadVariableOp/model_76/INJECTION_MASKS/BiasAdd/ReadVariableOp2`
.model_76/INJECTION_MASKS/MatMul/ReadVariableOp.model_76/INJECTION_MASKS/MatMul/ReadVariableOp2V
)model_76/conv1d_81/BiasAdd/ReadVariableOp)model_76/conv1d_81/BiasAdd/ReadVariableOp2n
5model_76/conv1d_81/Conv1D/ExpandDims_1/ReadVariableOp5model_76/conv1d_81/Conv1D/ExpandDims_1/ReadVariableOp2V
)model_76/conv1d_82/BiasAdd/ReadVariableOp)model_76/conv1d_82/BiasAdd/ReadVariableOp2n
5model_76/conv1d_82/Conv1D/ExpandDims_1/ReadVariableOp5model_76/conv1d_82/Conv1D/ExpandDims_1/ReadVariableOp2V
)model_76/conv1d_83/BiasAdd/ReadVariableOp)model_76/conv1d_83/BiasAdd/ReadVariableOp2n
5model_76/conv1d_83/Conv1D/ExpandDims_1/ReadVariableOp5model_76/conv1d_83/Conv1D/ExpandDims_1/ReadVariableOp2T
(model_76/dense_89/BiasAdd/ReadVariableOp(model_76/dense_89/BiasAdd/ReadVariableOp2X
*model_76/dense_89/Tensordot/ReadVariableOp*model_76/dense_89/Tensordot/ReadVariableOp:VR
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
У

e
F__inference_dropout_83_layer_call_and_return_conditional_losses_799759

inputs
identityR
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *6Щ?h
dropout/MulMulinputsdropout/Const:output:0*
T0*+
_output_shapes
:џџџџџџџџџ2yQ
dropout/ShapeShapeinputs*
T0*
_output_shapes
::эЯ
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*+
_output_shapes
:џџџџџџџџџ2y*
dtype0*
seedш[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 * KК>Њ
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*+
_output_shapes
:џџџџџџџџџ2yT
dropout/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    
dropout/SelectV2SelectV2dropout/GreaterEqual:z:0dropout/Mul:z:0dropout/Const_1:output:0*
T0*+
_output_shapes
:џџџџџџџџџ2ye
IdentityIdentitydropout/SelectV2:output:0*
T0*+
_output_shapes
:џџџџџџџџџ2y"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:џџџџџџџџџ2y:S O
+
_output_shapes
:џџџџџџџџџ2y
 
_user_specified_nameinputs
щ

*__inference_conv1d_81_layer_call_fn_799721

inputs
unknown:Fy
	unknown_0:y
identityЂStatefulPartitionedCallэ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:џџџџџџџџџ2y*$
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
E__inference_conv1d_81_layer_call_and_return_conditional_losses_799009s
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:џџџџџџџџџ2y`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:џџџџџџџџџ: : 22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
Ъ

E__inference_conv1d_81_layer_call_and_return_conditional_losses_799737

inputsA
+conv1d_expanddims_1_readvariableop_resource:Fy-
biasadd_readvariableop_resource:y
identityЂBiasAdd/ReadVariableOpЂ"Conv1D/ExpandDims_1/ReadVariableOp`
Conv1D/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
§џџџџџџџџ
Conv1D/ExpandDims
ExpandDimsinputsConv1D/ExpandDims/dim:output:0*
T0*0
_output_shapes
:џџџџџџџџџ
"Conv1D/ExpandDims_1/ReadVariableOpReadVariableOp+conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:Fy*
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
:FyЌ
Conv1DConv2DConv1D/ExpandDims:output:0Conv1D/ExpandDims_1:output:0*
T0*/
_output_shapes
:џџџџџџџџџ2y*
paddingSAME*
strides
)
Conv1D/SqueezeSqueezeConv1D:output:0*
T0*+
_output_shapes
:џџџџџџџџџ2y*
squeeze_dims

§џџџџџџџџr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:y*
dtype0
BiasAddBiasAddConv1D/Squeeze:output:0BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:џџџџџџџџџ2yZ
SigmoidSigmoidBiasAdd:output:0*
T0*+
_output_shapes
:џџџџџџџџџ2y^
IdentityIdentitySigmoid:y:0^NoOp*
T0*+
_output_shapes
:џџџџџџџџџ2y
NoOpNoOp^BiasAdd/ReadVariableOp#^Conv1D/ExpandDims_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:џџџџџџџџџ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2H
"Conv1D/ExpandDims_1/ReadVariableOp"Conv1D/ExpandDims_1/ReadVariableOp:T P
,
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
И
O
#__inference__update_step_xla_286536
gradient
variable:k*
_XlaMustCompile(*(
_construction_contextkEagerRuntime*
_input_shapes
:k: *
	_noinline(:($
"
_user_specified_name
variable:H D

_output_shapes

:k
"
_user_specified_name
gradient
Р
G
+__inference_dropout_83_layer_call_fn_799747

inputs
identityФ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:џџџџџџџџџ2y* 
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
F__inference_dropout_83_layer_call_and_return_conditional_losses_799167d
IdentityIdentityPartitionedCall:output:0*
T0*+
_output_shapes
:џџџџџџџџџ2y"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:џџџџџџџџџ2y:S O
+
_output_shapes
:џџџџџџџџџ2y
 
_user_specified_nameinputs
о
_
C__inference_reshape_76_layer_call_and_return_conditional_losses_534

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
Ф
S
#__inference__update_step_xla_286516
gradient
variable:ja*
_XlaMustCompile(*(
_construction_contextkEagerRuntime*#
_input_shapes
:ja: *
	_noinline(:($
"
_user_specified_name
variable:L H
"
_output_shapes
:ja
"
_user_specified_name
gradient
ч

*__inference_conv1d_83_layer_call_fn_799846

inputs
unknown:sak
	unknown_0:k
identityЂStatefulPartitionedCallэ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:џџџџџџџџџk*$
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
E__inference_conv1d_83_layer_call_and_return_conditional_losses_799120s
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:џџџџџџџџџk`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:џџџџџџџџџa: : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:џџџџџџџџџa
 
_user_specified_nameinputs

d
+__inference_dropout_83_layer_call_fn_799742

inputs
identityЂStatefulPartitionedCallд
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:џџџџџџџџџ2y* 
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
F__inference_dropout_83_layer_call_and_return_conditional_losses_799027s
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:џџџџџџџџџ2y`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:џџџџџџџџџ2y22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:џџџџџџџџџ2y
 
_user_specified_nameinputs
щ
d
F__inference_dropout_83_layer_call_and_return_conditional_losses_799167

inputs

identity_1R
IdentityIdentityinputs*
T0*+
_output_shapes
:џџџџџџџџџ2y_

Identity_1IdentityIdentity:output:0*
T0*+
_output_shapes
:џџџџџџџџџ2y"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:џџџџџџџџџ2y:S O
+
_output_shapes
:џџџџџџџџџ2y
 
_user_specified_nameinputs
р
В
#__inference_internal_grad_fn_800201
result_grads_0
result_grads_1
result_grads_2
mul_conv1d_83_beta
mul_conv1d_83_biasadd
identity

identity_1|
mulMulmul_conv1d_83_betamul_conv1d_83_biasadd^result_grads_0*
T0*+
_output_shapes
:џџџџџџџџџkQ
SigmoidSigmoidmul:z:0*
T0*+
_output_shapes
:џџџџџџџџџkm
mul_1Mulmul_conv1d_83_betamul_conv1d_83_biasadd*
T0*+
_output_shapes
:џџџџџџџџџkJ
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?]
subSubsub/x:output:0Sigmoid:y:0*
T0*+
_output_shapes
:џџџџџџџџџkV
mul_2Mul	mul_1:z:0sub:z:0*
T0*+
_output_shapes
:џџџџџџџџџkJ
add/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?]
addAddV2add/x:output:0	mul_2:z:0*
T0*+
_output_shapes
:џџџџџџџџџkX
mul_3MulSigmoid:y:0add:z:0*
T0*+
_output_shapes
:џџџџџџџџџk]
SquareSquaremul_conv1d_83_biasadd*
T0*+
_output_shapes
:џџџџџџџџџk^
mul_4Mulresult_grads_0
Square:y:0*
T0*+
_output_shapes
:џџџџџџџџџkZ
mul_5Mul	mul_4:z:0Sigmoid:y:0*
T0*+
_output_shapes
:џџџџџџџџџkL
sub_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?a
sub_1Subsub_1/x:output:0Sigmoid:y:0*
T0*+
_output_shapes
:џџџџџџџџџkX
mul_6Mul	mul_5:z:0	sub_1:z:0*
T0*+
_output_shapes
:џџџџџџџџџkZ
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
:џџџџџџџџџkU
IdentityIdentity	mul_7:z:0*
T0*+
_output_shapes
:џџџџџџџџџkE

Identity_1IdentitySum:output:0*
T0*
_output_shapes
: "!

identity_1Identity_1:output:0"
identityIdentity:output:0*\
_input_shapesK
I:џџџџџџџџџk:џџџџџџџџџk: : :џџџџџџџџџk:1-
+
_output_shapes
:џџџџџџџџџk:
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
:џџџџџџџџџk
(
_user_specified_nameresult_grads_1: 
&
 _has_manual_control_dependencies(
+
_output_shapes
:џџџџџџџџџk
(
_user_specified_nameresult_grads_0
О
D
(__inference_reshape_76_layer_call_fn_733

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
C__inference_reshape_76_layer_call_and_return_conditional_losses_728e
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
Ё

ќ
K__inference_INJECTION_MASKS_layer_call_and_return_conditional_losses_799901

inputs0
matmul_readvariableop_resource:k-
biasadd_readvariableop_resource:
identityЂBiasAdd/ReadVariableOpЂMatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:k*
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
:џџџџџџџџџk: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:џџџџџџџџџk
 
_user_specified_nameinputs
р
В
#__inference_internal_grad_fn_800145
result_grads_0
result_grads_1
result_grads_2
mul_conv1d_83_beta
mul_conv1d_83_biasadd
identity

identity_1|
mulMulmul_conv1d_83_betamul_conv1d_83_biasadd^result_grads_0*
T0*+
_output_shapes
:џџџџџџџџџkQ
SigmoidSigmoidmul:z:0*
T0*+
_output_shapes
:џџџџџџџџџkm
mul_1Mulmul_conv1d_83_betamul_conv1d_83_biasadd*
T0*+
_output_shapes
:џџџџџџџџџkJ
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?]
subSubsub/x:output:0Sigmoid:y:0*
T0*+
_output_shapes
:џџџџџџџџџkV
mul_2Mul	mul_1:z:0sub:z:0*
T0*+
_output_shapes
:џџџџџџџџџkJ
add/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?]
addAddV2add/x:output:0	mul_2:z:0*
T0*+
_output_shapes
:џџџџџџџџџkX
mul_3MulSigmoid:y:0add:z:0*
T0*+
_output_shapes
:џџџџџџџџџk]
SquareSquaremul_conv1d_83_biasadd*
T0*+
_output_shapes
:џџџџџџџџџk^
mul_4Mulresult_grads_0
Square:y:0*
T0*+
_output_shapes
:џџџџџџџџџkZ
mul_5Mul	mul_4:z:0Sigmoid:y:0*
T0*+
_output_shapes
:џџџџџџџџџkL
sub_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?a
sub_1Subsub_1/x:output:0Sigmoid:y:0*
T0*+
_output_shapes
:џџџџџџџџџkX
mul_6Mul	mul_5:z:0	sub_1:z:0*
T0*+
_output_shapes
:џџџџџџџџџkZ
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
:џџџџџџџџџkU
IdentityIdentity	mul_7:z:0*
T0*+
_output_shapes
:џџџџџџџџџkE

Identity_1IdentitySum:output:0*
T0*
_output_shapes
: "!

identity_1Identity_1:output:0"
identityIdentity:output:0*\
_input_shapesK
I:џџџџџџџџџk:џџџџџџџџџk: : :џџџџџџџџџk:1-
+
_output_shapes
:џџџџџџџџџk:
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
:џџџџџџџџџk
(
_user_specified_nameresult_grads_1: 
&
 _has_manual_control_dependencies(
+
_output_shapes
:џџџџџџџџџk
(
_user_specified_nameresult_grads_0
і
k
O__inference_whiten_passthrough_38_layer_call_and_return_conditional_losses_1185

inputs
identityГ
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
 @E8 *&
f!R
__inference_crop_samples_1151I
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
ќ

E__inference_conv1d_82_layer_call_and_return_conditional_losses_799090

inputsA
+conv1d_expanddims_1_readvariableop_resource:ja-
biasadd_readvariableop_resource:a

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
:џџџџџџџџџ2
"Conv1D/ExpandDims_1/ReadVariableOpReadVariableOp+conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:ja*
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
:jaЌ
Conv1DConv2DConv1D/ExpandDims:output:0Conv1D/ExpandDims_1:output:0*
T0*/
_output_shapes
:џџџџџџџџџa*
paddingSAME*
strides
=
Conv1D/SqueezeSqueezeConv1D:output:0*
T0*+
_output_shapes
:џџџџџџџџџa*
squeeze_dims

§џџџџџџџџr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:a*
dtype0
BiasAddBiasAddConv1D/Squeeze:output:0BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:џџџџџџџџџaI
betaConst*
_output_shapes
: *
dtype0*
valueB
 *  ?a
mulMulbeta:output:0BiasAdd:output:0*
T0*+
_output_shapes
:џџџџџџџџџaQ
SigmoidSigmoidmul:z:0*
T0*+
_output_shapes
:џџџџџџџџџaa
mul_1MulBiasAdd:output:0Sigmoid:y:0*
T0*+
_output_shapes
:џџџџџџџџџaU
IdentityIdentity	mul_1:z:0*
T0*+
_output_shapes
:џџџџџџџџџaФ
	IdentityN	IdentityN	mul_1:z:0BiasAdd:output:0beta:output:0*
T
2*,
_gradient_op_typeCustomGradient-799081*D
_output_shapes2
0:џџџџџџџџџa:џџџџџџџџџa: g

Identity_1IdentityIdentityN:output:0^NoOp*
T0*+
_output_shapes
:џџџџџџџџџa
NoOpNoOp^BiasAdd/ReadVariableOp#^Conv1D/ExpandDims_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:џџџџџџџџџ2: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2H
"Conv1D/ExpandDims_1/ReadVariableOp"Conv1D/ExpandDims_1/ReadVariableOp:S O
+
_output_shapes
:џџџџџџџџџ2
 
_user_specified_nameinputs
Ќ
K
#__inference__update_step_xla_286541
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
ѕ

)__inference_model_76_layer_call_fn_799513
inputs_offsource
inputs_onsource
unknown:Fy
	unknown_0:y
	unknown_1:y
	unknown_2:
	unknown_3:ja
	unknown_4:a
	unknown_5:sak
	unknown_6:k
	unknown_7:k
	unknown_8:
identityЂStatefulPartitionedCallь
StatefulPartitionedCallStatefulPartitionedCallinputs_offsourceinputs_onsourceunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*,
_read_only_resource_inputs

	
*<
config_proto,*

CPU

GPU(2*0J

  zE8 *M
fHRF
D__inference_model_76_layer_call_and_return_conditional_losses_799289o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*X
_input_shapesG
E:џџџџџџџџџ:џџџџџџџџџ : : : : : : : : : : 22
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
ќ

E__inference_conv1d_83_layer_call_and_return_conditional_losses_799870

inputsA
+conv1d_expanddims_1_readvariableop_resource:sak-
biasadd_readvariableop_resource:k

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
:џџџџџџџџџa
"Conv1D/ExpandDims_1/ReadVariableOpReadVariableOp+conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:sak*
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
:sakЌ
Conv1DConv2DConv1D/ExpandDims:output:0Conv1D/ExpandDims_1:output:0*
T0*/
_output_shapes
:џџџџџџџџџk*
paddingSAME*
strides
C
Conv1D/SqueezeSqueezeConv1D:output:0*
T0*+
_output_shapes
:џџџџџџџџџk*
squeeze_dims

§џџџџџџџџr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:k*
dtype0
BiasAddBiasAddConv1D/Squeeze:output:0BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:џџџџџџџџџkI
betaConst*
_output_shapes
: *
dtype0*
valueB
 *  ?a
mulMulbeta:output:0BiasAdd:output:0*
T0*+
_output_shapes
:џџџџџџџџџkQ
SigmoidSigmoidmul:z:0*
T0*+
_output_shapes
:џџџџџџџџџka
mul_1MulBiasAdd:output:0Sigmoid:y:0*
T0*+
_output_shapes
:џџџџџџџџџkU
IdentityIdentity	mul_1:z:0*
T0*+
_output_shapes
:џџџџџџџџџkФ
	IdentityN	IdentityN	mul_1:z:0BiasAdd:output:0beta:output:0*
T
2*,
_gradient_op_typeCustomGradient-799861*D
_output_shapes2
0:џџџџџџџџџk:џџџџџџџџџk: g

Identity_1IdentityIdentityN:output:0^NoOp*
T0*+
_output_shapes
:џџџџџџџџџk
NoOpNoOp^BiasAdd/ReadVariableOp#^Conv1D/ExpandDims_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:џџџџџџџџџa: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2H
"Conv1D/ExpandDims_1/ReadVariableOp"Conv1D/ExpandDims_1/ReadVariableOp:S O
+
_output_shapes
:џџџџџџџџџa
 
_user_specified_nameinputs
А
ћ
D__inference_dense_89_layer_call_and_return_conditional_losses_799060

inputs3
!tensordot_readvariableop_resource:y-
biasadd_readvariableop_resource:
identityЂBiasAdd/ReadVariableOpЂTensordot/ReadVariableOpz
Tensordot/ReadVariableOpReadVariableOp!tensordot_readvariableop_resource*
_output_shapes

:y*
dtype0X
Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:_
Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       S
Tensordot/ShapeShapeinputs*
T0*
_output_shapes
::эЯY
Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : Л
Tensordot/GatherV2GatherV2Tensordot/Shape:output:0Tensordot/free:output:0 Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:[
Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : П
Tensordot/GatherV2_1GatherV2Tensordot/Shape:output:0Tensordot/axes:output:0"Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:Y
Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: n
Tensordot/ProdProdTensordot/GatherV2:output:0Tensordot/Const:output:0*
T0*
_output_shapes
: [
Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: t
Tensordot/Prod_1ProdTensordot/GatherV2_1:output:0Tensordot/Const_1:output:0*
T0*
_output_shapes
: W
Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 
Tensordot/concatConcatV2Tensordot/free:output:0Tensordot/axes:output:0Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:y
Tensordot/stackPackTensordot/Prod:output:0Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:y
Tensordot/transpose	TransposeinputsTensordot/concat:output:0*
T0*+
_output_shapes
:џџџџџџџџџ2y
Tensordot/ReshapeReshapeTensordot/transpose:y:0Tensordot/stack:output:0*
T0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ
Tensordot/MatMulMatMulTensordot/Reshape:output:0 Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ[
Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:Y
Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : Ї
Tensordot/concat_1ConcatV2Tensordot/GatherV2:output:0Tensordot/Const_2:output:0 Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:
	TensordotReshapeTensordot/MatMul:product:0Tensordot/concat_1:output:0*
T0*+
_output_shapes
:џџџџџџџџџ2r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0|
BiasAddBiasAddTensordot:output:0BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:џџџџџџџџџ2Z
SigmoidSigmoidBiasAdd:output:0*
T0*+
_output_shapes
:џџџџџџџџџ2^
IdentityIdentitySigmoid:y:0^NoOp*
T0*+
_output_shapes
:џџџџџџџџџ2z
NoOpNoOp^BiasAdd/ReadVariableOp^Tensordot/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:џџџџџџџџџ2y: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp24
Tensordot/ReadVariableOpTensordot/ReadVariableOp:S O
+
_output_shapes
:џџџџџџџџџ2y
 
_user_specified_nameinputs
ч

*__inference_conv1d_82_layer_call_fn_799813

inputs
unknown:ja
	unknown_0:a
identityЂStatefulPartitionedCallэ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:џџџџџџџџџa*$
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
E__inference_conv1d_82_layer_call_and_return_conditional_losses_799090s
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:џџџџџџџџџa`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:џџџџџџџџџ2: : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:џџџџџџџџџ2
 
_user_specified_nameinputs
О
b
F__inference_flatten_76_layer_call_and_return_conditional_losses_799132

inputs
identityV
ConstConst*
_output_shapes
:*
dtype0*
valueB"џџџџk   \
ReshapeReshapeinputsConst:output:0*
T0*'
_output_shapes
:џџџџџџџџџkX
IdentityIdentityReshape:output:0*
T0*'
_output_shapes
:џџџџџџџџџk"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:џџџџџџџџџk:S O
+
_output_shapes
:џџџџџџџџџk
 
_user_specified_nameinputs
Ў
E
)__inference_restored_function_body_285667

inputs
identityЏ
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
  zE8 *X
fSRQ
O__inference_whiten_passthrough_38_layer_call_and_return_conditional_losses_1335e
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


#__inference_internal_grad_fn_800061
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
:џџџџџџџџџaQ
SigmoidSigmoidmul:z:0*
T0*+
_output_shapes
:џџџџџџџџџaY
mul_1Mulmul_betamul_biasadd*
T0*+
_output_shapes
:џџџџџџџџџaJ
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?]
subSubsub/x:output:0Sigmoid:y:0*
T0*+
_output_shapes
:џџџџџџџџџaV
mul_2Mul	mul_1:z:0sub:z:0*
T0*+
_output_shapes
:џџџџџџџџџaJ
add/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?]
addAddV2add/x:output:0	mul_2:z:0*
T0*+
_output_shapes
:џџџџџџџџџaX
mul_3MulSigmoid:y:0add:z:0*
T0*+
_output_shapes
:џџџџџџџџџaS
SquareSquaremul_biasadd*
T0*+
_output_shapes
:џџџџџџџџџa^
mul_4Mulresult_grads_0
Square:y:0*
T0*+
_output_shapes
:џџџџџџџџџaZ
mul_5Mul	mul_4:z:0Sigmoid:y:0*
T0*+
_output_shapes
:џџџџџџџџџaL
sub_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?a
sub_1Subsub_1/x:output:0Sigmoid:y:0*
T0*+
_output_shapes
:џџџџџџџџџaX
mul_6Mul	mul_5:z:0	sub_1:z:0*
T0*+
_output_shapes
:џџџџџџџџџaZ
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
:џџџџџџџџџaU
IdentityIdentity	mul_7:z:0*
T0*+
_output_shapes
:џџџџџџџџџaE

Identity_1IdentitySum:output:0*
T0*
_output_shapes
: "!

identity_1Identity_1:output:0"
identityIdentity:output:0*\
_input_shapesK
I:џџџџџџџџџa:џџџџџџџџџa: : :џџџџџџџџџa:1-
+
_output_shapes
:џџџџџџџџџa:
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
:џџџџџџџџџa
(
_user_specified_nameresult_grads_1: 
&
 _has_manual_control_dependencies(
+
_output_shapes
:џџџџџџџџџa
(
_user_specified_nameresult_grads_0
О
b
F__inference_flatten_76_layer_call_and_return_conditional_losses_799881

inputs
identityV
ConstConst*
_output_shapes
:*
dtype0*
valueB"џџџџk   \
ReshapeReshapeinputsConst:output:0*
T0*'
_output_shapes
:џџџџџџџџџkX
IdentityIdentityReshape:output:0*
T0*'
_output_shapes
:џџџџџџџџџk"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:џџџџџџџџџk:S O
+
_output_shapes
:џџџџџџџџџk
 
_user_specified_nameinputs
И
G
+__inference_flatten_76_layer_call_fn_799875

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
:џџџџџџџџџk* 
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
F__inference_flatten_76_layer_call_and_return_conditional_losses_799132`
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:џџџџџџџџџk"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:џџџџџџџџџk:S O
+
_output_shapes
:џџџџџџџџџk
 
_user_specified_nameinputs
ќ

E__inference_conv1d_83_layer_call_and_return_conditional_losses_799120

inputsA
+conv1d_expanddims_1_readvariableop_resource:sak-
biasadd_readvariableop_resource:k

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
:џџџџџџџџџa
"Conv1D/ExpandDims_1/ReadVariableOpReadVariableOp+conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:sak*
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
:sakЌ
Conv1DConv2DConv1D/ExpandDims:output:0Conv1D/ExpandDims_1:output:0*
T0*/
_output_shapes
:џџџџџџџџџk*
paddingSAME*
strides
C
Conv1D/SqueezeSqueezeConv1D:output:0*
T0*+
_output_shapes
:џџџџџџџџџk*
squeeze_dims

§џџџџџџџџr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:k*
dtype0
BiasAddBiasAddConv1D/Squeeze:output:0BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:џџџџџџџџџkI
betaConst*
_output_shapes
: *
dtype0*
valueB
 *  ?a
mulMulbeta:output:0BiasAdd:output:0*
T0*+
_output_shapes
:џџџџџџџџџkQ
SigmoidSigmoidmul:z:0*
T0*+
_output_shapes
:џџџџџџџџџka
mul_1MulBiasAdd:output:0Sigmoid:y:0*
T0*+
_output_shapes
:џџџџџџџџџkU
IdentityIdentity	mul_1:z:0*
T0*+
_output_shapes
:џџџџџџџџџkФ
	IdentityN	IdentityN	mul_1:z:0BiasAdd:output:0beta:output:0*
T
2*,
_gradient_op_typeCustomGradient-799111*D
_output_shapes2
0:џџџџџџџџџk:џџџџџџџџџk: g

Identity_1IdentityIdentityN:output:0^NoOp*
T0*+
_output_shapes
:џџџџџџџџџk
NoOpNoOp^BiasAdd/ReadVariableOp#^Conv1D/ExpandDims_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:џџџџџџџџџa: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2H
"Conv1D/ExpandDims_1/ReadVariableOp"Conv1D/ExpandDims_1/ReadVariableOp:S O
+
_output_shapes
:џџџџџџџџџa
 
_user_specified_nameinputs
п

0__inference_INJECTION_MASKS_layer_call_fn_799890

inputs
unknown:k
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
K__inference_INJECTION_MASKS_layer_call_and_return_conditional_losses_799145o
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
:џџџџџџџџџk: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:џџџџџџџџџk
 
_user_specified_nameinputs
Ќ
K
#__inference__update_step_xla_286531
gradient
variable:k*
_XlaMustCompile(*(
_construction_contextkEagerRuntime*
_input_shapes

:k: *
	_noinline(:($
"
_user_specified_name
variable:D @

_output_shapes
:k
"
_user_specified_name
gradient
Ќ
K
#__inference__update_step_xla_286521
gradient
variable:a*
_XlaMustCompile(*(
_construction_contextkEagerRuntime*
_input_shapes

:a: *
	_noinline(:($
"
_user_specified_name
variable:D @

_output_shapes
:a
"
_user_specified_name
gradient


#__inference_internal_grad_fn_800033
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
:џџџџџџџџџkQ
SigmoidSigmoidmul:z:0*
T0*+
_output_shapes
:џџџџџџџџџkY
mul_1Mulmul_betamul_biasadd*
T0*+
_output_shapes
:џџџџџџџџџkJ
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?]
subSubsub/x:output:0Sigmoid:y:0*
T0*+
_output_shapes
:џџџџџџџџџkV
mul_2Mul	mul_1:z:0sub:z:0*
T0*+
_output_shapes
:џџџџџџџџџkJ
add/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?]
addAddV2add/x:output:0	mul_2:z:0*
T0*+
_output_shapes
:џџџџџџџџџkX
mul_3MulSigmoid:y:0add:z:0*
T0*+
_output_shapes
:џџџџџџџџџkS
SquareSquaremul_biasadd*
T0*+
_output_shapes
:џџџџџџџџџk^
mul_4Mulresult_grads_0
Square:y:0*
T0*+
_output_shapes
:џџџџџџџџџkZ
mul_5Mul	mul_4:z:0Sigmoid:y:0*
T0*+
_output_shapes
:џџџџџџџџџkL
sub_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?a
sub_1Subsub_1/x:output:0Sigmoid:y:0*
T0*+
_output_shapes
:џџџџџџџџџkX
mul_6Mul	mul_5:z:0	sub_1:z:0*
T0*+
_output_shapes
:џџџџџџџџџkZ
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
:џџџџџџџџџkU
IdentityIdentity	mul_7:z:0*
T0*+
_output_shapes
:џџџџџџџџџkE

Identity_1IdentitySum:output:0*
T0*
_output_shapes
: "!

identity_1Identity_1:output:0"
identityIdentity:output:0*\
_input_shapesK
I:џџџџџџџџџk:џџџџџџџџџk: : :џџџџџџџџџk:1-
+
_output_shapes
:џџџџџџџџџk:
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
:џџџџџџџџџk
(
_user_specified_nameresult_grads_1: 
&
 _has_manual_control_dependencies(
+
_output_shapes
:џџџџџџџџџk
(
_user_specified_nameresult_grads_0
і
k
O__inference_whiten_passthrough_38_layer_call_and_return_conditional_losses_1335

inputs
identityГ
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
 @E8 *&
f!R
__inference_crop_samples_1151I
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


#__inference_internal_grad_fn_800089
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
:џџџџџџџџџaQ
SigmoidSigmoidmul:z:0*
T0*+
_output_shapes
:џџџџџџџџџaY
mul_1Mulmul_betamul_biasadd*
T0*+
_output_shapes
:џџџџџџџџџaJ
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?]
subSubsub/x:output:0Sigmoid:y:0*
T0*+
_output_shapes
:џџџџџџџџџaV
mul_2Mul	mul_1:z:0sub:z:0*
T0*+
_output_shapes
:џџџџџџџџџaJ
add/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?]
addAddV2add/x:output:0	mul_2:z:0*
T0*+
_output_shapes
:џџџџџџџџџaX
mul_3MulSigmoid:y:0add:z:0*
T0*+
_output_shapes
:џџџџџџџџџaS
SquareSquaremul_biasadd*
T0*+
_output_shapes
:џџџџџџџџџa^
mul_4Mulresult_grads_0
Square:y:0*
T0*+
_output_shapes
:џџџџџџџџџaZ
mul_5Mul	mul_4:z:0Sigmoid:y:0*
T0*+
_output_shapes
:џџџџџџџџџaL
sub_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?a
sub_1Subsub_1/x:output:0Sigmoid:y:0*
T0*+
_output_shapes
:џџџџџџџџџaX
mul_6Mul	mul_5:z:0	sub_1:z:0*
T0*+
_output_shapes
:џџџџџџџџџaZ
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
:џџџџџџџџџaU
IdentityIdentity	mul_7:z:0*
T0*+
_output_shapes
:џџџџџџџџџaE

Identity_1IdentitySum:output:0*
T0*
_output_shapes
: "!

identity_1Identity_1:output:0"
identityIdentity:output:0*\
_input_shapesK
I:џџџџџџџџџa:џџџџџџџџџa: : :џџџџџџџџџa:1-
+
_output_shapes
:џџџџџџџџџa:
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
:џџџџџџџџџa
(
_user_specified_nameresult_grads_1: 
&
 _has_manual_control_dependencies(
+
_output_shapes
:џџџџџџџџџa
(
_user_specified_nameresult_grads_0
о
_
C__inference_reshape_76_layer_call_and_return_conditional_losses_728

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
У

e
F__inference_dropout_83_layer_call_and_return_conditional_losses_799027

inputs
identityR
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *6Щ?h
dropout/MulMulinputsdropout/Const:output:0*
T0*+
_output_shapes
:џџџџџџџџџ2yQ
dropout/ShapeShapeinputs*
T0*
_output_shapes
::эЯ
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*+
_output_shapes
:џџџџџџџџџ2y*
dtype0*
seedш[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 * KК>Њ
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*+
_output_shapes
:џџџџџџџџџ2yT
dropout/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    
dropout/SelectV2SelectV2dropout/GreaterEqual:z:0dropout/Mul:z:0dropout/Const_1:output:0*
T0*+
_output_shapes
:џџџџџџџџџ2ye
IdentityIdentitydropout/SelectV2:output:0*
T0*+
_output_shapes
:џџџџџџџџџ2y"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:џџџџџџџџџ2y:S O
+
_output_shapes
:џџџџџџџџџ2y
 
_user_specified_nameinputs
И
O
#__inference__update_step_xla_286506
gradient
variable:y*
_XlaMustCompile(*(
_construction_contextkEagerRuntime*
_input_shapes
:y: *
	_noinline(:($
"
_user_specified_name
variable:H D

_output_shapes

:y
"
_user_specified_name
gradient
ѕ

)__inference_model_76_layer_call_fn_799487
inputs_offsource
inputs_onsource
unknown:Fy
	unknown_0:y
	unknown_1:y
	unknown_2:
	unknown_3:ja
	unknown_4:a
	unknown_5:sak
	unknown_6:k
	unknown_7:k
	unknown_8:
identityЂStatefulPartitionedCallь
StatefulPartitionedCallStatefulPartitionedCallinputs_offsourceinputs_onsourceunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*,
_read_only_resource_inputs

	
*<
config_proto,*

CPU

GPU(2*0J

  zE8 *M
fHRF
D__inference_model_76_layer_call_and_return_conditional_losses_799229o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*X
_input_shapesG
E:џџџџџџџџџ:џџџџџџџџџ : : : : : : : : : : 22
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
щ
d
F__inference_dropout_83_layer_call_and_return_conditional_losses_799764

inputs

identity_1R
IdentityIdentityinputs*
T0*+
_output_shapes
:џџџџџџџџџ2y_

Identity_1IdentityIdentity:output:0*
T0*+
_output_shapes
:џџџџџџџџџ2y"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:џџџџџџџџџ2y:S O
+
_output_shapes
:џџџџџџџџџ2y
 
_user_specified_nameinputs
и
P
4__inference_whiten_passthrough_38_layer_call_fn_1190

inputs
identityЮ
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
 @E8 *X
fSRQ
O__inference_whiten_passthrough_38_layer_call_and_return_conditional_losses_1185e
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
#__inference_internal_grad_fn_800173
result_grads_0
result_grads_1
result_grads_2
mul_conv1d_82_beta
mul_conv1d_82_biasadd
identity

identity_1|
mulMulmul_conv1d_82_betamul_conv1d_82_biasadd^result_grads_0*
T0*+
_output_shapes
:џџџџџџџџџaQ
SigmoidSigmoidmul:z:0*
T0*+
_output_shapes
:џџџџџџџџџam
mul_1Mulmul_conv1d_82_betamul_conv1d_82_biasadd*
T0*+
_output_shapes
:џџџџџџџџџaJ
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?]
subSubsub/x:output:0Sigmoid:y:0*
T0*+
_output_shapes
:џџџџџџџџџaV
mul_2Mul	mul_1:z:0sub:z:0*
T0*+
_output_shapes
:џџџџџџџџџaJ
add/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?]
addAddV2add/x:output:0	mul_2:z:0*
T0*+
_output_shapes
:џџџџџџџџџaX
mul_3MulSigmoid:y:0add:z:0*
T0*+
_output_shapes
:џџџџџџџџџa]
SquareSquaremul_conv1d_82_biasadd*
T0*+
_output_shapes
:џџџџџџџџџa^
mul_4Mulresult_grads_0
Square:y:0*
T0*+
_output_shapes
:џџџџџџџџџaZ
mul_5Mul	mul_4:z:0Sigmoid:y:0*
T0*+
_output_shapes
:џџџџџџџџџaL
sub_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?a
sub_1Subsub_1/x:output:0Sigmoid:y:0*
T0*+
_output_shapes
:џџџџџџџџџaX
mul_6Mul	mul_5:z:0	sub_1:z:0*
T0*+
_output_shapes
:џџџџџџџџџaZ
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
:џџџџџџџџџaU
IdentityIdentity	mul_7:z:0*
T0*+
_output_shapes
:џџџџџџџџџaE

Identity_1IdentitySum:output:0*
T0*
_output_shapes
: "!

identity_1Identity_1:output:0"
identityIdentity:output:0*\
_input_shapesK
I:џџџџџџџџџa:џџџџџџџџџa: : :џџџџџџџџџa:1-
+
_output_shapes
:џџџџџџџџџa:
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
:џџџџџџџџџa
(
_user_specified_nameresult_grads_1: 
&
 _has_manual_control_dependencies(
+
_output_shapes
:џџџџџџџџџa
(
_user_specified_nameresult_grads_0
Ю)

D__inference_model_76_layer_call_and_return_conditional_losses_799191
	offsource
onsource&
conv1d_81_799158:Fy
conv1d_81_799160:y!
dense_89_799169:y
dense_89_799171:&
conv1d_82_799174:ja
conv1d_82_799176:a&
conv1d_83_799179:sak
conv1d_83_799181:k(
injection_masks_799185:k$
injection_masks_799187:
identityЂ'INJECTION_MASKS/StatefulPartitionedCallЂ!conv1d_81/StatefulPartitionedCallЂ!conv1d_82/StatefulPartitionedCallЂ!conv1d_83/StatefulPartitionedCallЂ dense_89/StatefulPartitionedCallС
%whiten_passthrough_38/PartitionedCallPartitionedCall	offsource*
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
)__inference_restored_function_body_285667л
reshape_76/PartitionedCallPartitionedCall.whiten_passthrough_38/PartitionedCall:output:0*
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
)__inference_restored_function_body_285673Є
!conv1d_81/StatefulPartitionedCallStatefulPartitionedCall#reshape_76/PartitionedCall:output:0conv1d_81_799158conv1d_81_799160*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:џџџџџџџџџ2y*$
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
E__inference_conv1d_81_layer_call_and_return_conditional_losses_799009ѓ
dropout_83/PartitionedCallPartitionedCall*conv1d_81/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:џџџџџџџџџ2y* 
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
F__inference_dropout_83_layer_call_and_return_conditional_losses_799167 
 dense_89/StatefulPartitionedCallStatefulPartitionedCall#dropout_83/PartitionedCall:output:0dense_89_799169dense_89_799171*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:џџџџџџџџџ2*$
_read_only_resource_inputs
*<
config_proto,*

CPU

GPU(2*0J

  zE8 *M
fHRF
D__inference_dense_89_layer_call_and_return_conditional_losses_799060Њ
!conv1d_82/StatefulPartitionedCallStatefulPartitionedCall)dense_89/StatefulPartitionedCall:output:0conv1d_82_799174conv1d_82_799176*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:џџџџџџџџџa*$
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
E__inference_conv1d_82_layer_call_and_return_conditional_losses_799090Ћ
!conv1d_83/StatefulPartitionedCallStatefulPartitionedCall*conv1d_82/StatefulPartitionedCall:output:0conv1d_83_799179conv1d_83_799181*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:џџџџџџџџџk*$
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
E__inference_conv1d_83_layer_call_and_return_conditional_losses_799120я
flatten_76/PartitionedCallPartitionedCall*conv1d_83/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџk* 
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
F__inference_flatten_76_layer_call_and_return_conditional_losses_799132И
'INJECTION_MASKS/StatefulPartitionedCallStatefulPartitionedCall#flatten_76/PartitionedCall:output:0injection_masks_799185injection_masks_799187*
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
K__inference_INJECTION_MASKS_layer_call_and_return_conditional_losses_799145
IdentityIdentity0INJECTION_MASKS/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџџ
NoOpNoOp(^INJECTION_MASKS/StatefulPartitionedCall"^conv1d_81/StatefulPartitionedCall"^conv1d_82/StatefulPartitionedCall"^conv1d_83/StatefulPartitionedCall!^dense_89/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*X
_input_shapesG
E:џџџџџџџџџ:џџџџџџџџџ : : : : : : : : : : 2R
'INJECTION_MASKS/StatefulPartitionedCall'INJECTION_MASKS/StatefulPartitionedCall2F
!conv1d_81/StatefulPartitionedCall!conv1d_81/StatefulPartitionedCall2F
!conv1d_82/StatefulPartitionedCall!conv1d_82/StatefulPartitionedCall2F
!conv1d_83/StatefulPartitionedCall!conv1d_83/StatefulPartitionedCall2D
 dense_89/StatefulPartitionedCall dense_89/StatefulPartitionedCall:VR
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
Ъ

E__inference_conv1d_81_layer_call_and_return_conditional_losses_799009

inputsA
+conv1d_expanddims_1_readvariableop_resource:Fy-
biasadd_readvariableop_resource:y
identityЂBiasAdd/ReadVariableOpЂ"Conv1D/ExpandDims_1/ReadVariableOp`
Conv1D/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
§џџџџџџџџ
Conv1D/ExpandDims
ExpandDimsinputsConv1D/ExpandDims/dim:output:0*
T0*0
_output_shapes
:џџџџџџџџџ
"Conv1D/ExpandDims_1/ReadVariableOpReadVariableOp+conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:Fy*
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
:FyЌ
Conv1DConv2DConv1D/ExpandDims:output:0Conv1D/ExpandDims_1:output:0*
T0*/
_output_shapes
:џџџџџџџџџ2y*
paddingSAME*
strides
)
Conv1D/SqueezeSqueezeConv1D:output:0*
T0*+
_output_shapes
:џџџџџџџџџ2y*
squeeze_dims

§џџџџџџџџr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:y*
dtype0
BiasAddBiasAddConv1D/Squeeze:output:0BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:џџџџџџџџџ2yZ
SigmoidSigmoidBiasAdd:output:0*
T0*+
_output_shapes
:џџџџџџџџџ2y^
IdentityIdentitySigmoid:y:0^NoOp*
T0*+
_output_shapes
:џџџџџџџџџ2y
NoOpNoOp^BiasAdd/ReadVariableOp#^Conv1D/ExpandDims_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:џџџџџџџџџ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2H
"Conv1D/ExpandDims_1/ReadVariableOp"Conv1D/ExpandDims_1/ReadVariableOp:T P
,
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
Ф
S
#__inference__update_step_xla_286496
gradient
variable:Fy*
_XlaMustCompile(*(
_construction_contextkEagerRuntime*#
_input_shapes
:Fy: *
	_noinline(:($
"
_user_specified_name
variable:L H
"
_output_shapes
:Fy
"
_user_specified_name
gradient<
#__inference_internal_grad_fn_800005CustomGradient-799861<
#__inference_internal_grad_fn_800033CustomGradient-799111<
#__inference_internal_grad_fn_800061CustomGradient-799828<
#__inference_internal_grad_fn_800089CustomGradient-799081<
#__inference_internal_grad_fn_800117CustomGradient-799674<
#__inference_internal_grad_fn_800145CustomGradient-799694<
#__inference_internal_grad_fn_800173CustomGradient-799578<
#__inference_internal_grad_fn_800201CustomGradient-799598<
#__inference_internal_grad_fn_800229CustomGradient-798948<
#__inference_internal_grad_fn_800257CustomGradient-798968"ѓ
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
StatefulPartitionedCall:0џџџџџџџџџtensorflow/serving/predict:ГИ

layer-0
layer-1
layer-2
layer_with_weights-0
layer-3
layer-4
layer_with_weights-1
layer-5
layer_with_weights-2
layer-6
layer_with_weights-3
layer-7
	layer-8

layer-9
layer_with_weights-4
layer-10
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
_default_save_signature
	optimizer

signatures
#_self_saveable_object_factories"
_tf_keras_network
D
#_self_saveable_object_factories"
_tf_keras_input_layer
Ъ
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
#_self_saveable_object_factories"
_tf_keras_layer
Ъ
	variables
trainable_variables
 regularization_losses
!	keras_api
"__call__
*#&call_and_return_all_conditional_losses
#$_self_saveable_object_factories"
_tf_keras_layer

%	variables
&trainable_variables
'regularization_losses
(	keras_api
)__call__
**&call_and_return_all_conditional_losses

+kernel
,bias
#-_self_saveable_object_factories
 ._jit_compiled_convolution_op"
_tf_keras_layer
с
/	variables
0trainable_variables
1regularization_losses
2	keras_api
3__call__
*4&call_and_return_all_conditional_losses
5_random_generator
#6_self_saveable_object_factories"
_tf_keras_layer
р
7	variables
8trainable_variables
9regularization_losses
:	keras_api
;__call__
*<&call_and_return_all_conditional_losses

=kernel
>bias
#?_self_saveable_object_factories"
_tf_keras_layer

@	variables
Atrainable_variables
Bregularization_losses
C	keras_api
D__call__
*E&call_and_return_all_conditional_losses

Fkernel
Gbias
#H_self_saveable_object_factories
 I_jit_compiled_convolution_op"
_tf_keras_layer

J	variables
Ktrainable_variables
Lregularization_losses
M	keras_api
N__call__
*O&call_and_return_all_conditional_losses

Pkernel
Qbias
#R_self_saveable_object_factories
 S_jit_compiled_convolution_op"
_tf_keras_layer
Ъ
T	variables
Utrainable_variables
Vregularization_losses
W	keras_api
X__call__
*Y&call_and_return_all_conditional_losses
#Z_self_saveable_object_factories"
_tf_keras_layer
D
#[_self_saveable_object_factories"
_tf_keras_input_layer
р
\	variables
]trainable_variables
^regularization_losses
_	keras_api
`__call__
*a&call_and_return_all_conditional_losses

bkernel
cbias
#d_self_saveable_object_factories"
_tf_keras_layer
f
+0
,1
=2
>3
F4
G5
P6
Q7
b8
c9"
trackable_list_wrapper
f
+0
,1
=2
>3
F4
G5
P6
Q7
b8
c9"
trackable_list_wrapper
 "
trackable_list_wrapper
Ъ
enon_trainable_variables

flayers
gmetrics
hlayer_regularization_losses
ilayer_metrics
	variables
trainable_variables
regularization_losses
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
Я
jtrace_0
ktrace_1
ltrace_2
mtrace_32ф
)__inference_model_76_layer_call_fn_799252
)__inference_model_76_layer_call_fn_799312
)__inference_model_76_layer_call_fn_799487
)__inference_model_76_layer_call_fn_799513Е
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
 zjtrace_0zktrace_1zltrace_2zmtrace_3
Л
ntrace_0
otrace_1
ptrace_2
qtrace_32а
D__inference_model_76_layer_call_and_return_conditional_losses_799152
D__inference_model_76_layer_call_and_return_conditional_losses_799191
D__inference_model_76_layer_call_and_return_conditional_losses_799616
D__inference_model_76_layer_call_and_return_conditional_losses_799712Е
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
 zntrace_0zotrace_1zptrace_2zqtrace_3
иBе
!__inference__wrapped_model_798986	OFFSOURCEONSOURCE"
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
r
_variables
s_iterations
t_learning_rate
u_index_dict
v
_momentums
w_velocities
x_update_step_xla"
experimentalOptimizer
,
yserving_default"
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
znon_trainable_variables

{layers
|metrics
}layer_regularization_losses
~layer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
ю
trace_02б
4__inference_whiten_passthrough_38_layer_call_fn_1190
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
 ztrace_0

trace_02ь
O__inference_whiten_passthrough_38_layer_call_and_return_conditional_losses_1335
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
 ztrace_0
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
В
non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
	variables
trainable_variables
 regularization_losses
"__call__
*#&call_and_return_all_conditional_losses
&#"call_and_return_conditional_losses"
_generic_user_object
ф
trace_02Х
(__inference_reshape_76_layer_call_fn_733
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
 ztrace_0
џ
trace_02р
C__inference_reshape_76_layer_call_and_return_conditional_losses_534
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
 ztrace_0
 "
trackable_dict_wrapper
.
+0
,1"
trackable_list_wrapper
.
+0
,1"
trackable_list_wrapper
 "
trackable_list_wrapper
В
non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
%	variables
&trainable_variables
'regularization_losses
)__call__
**&call_and_return_all_conditional_losses
&*"call_and_return_conditional_losses"
_generic_user_object
ц
trace_02Ч
*__inference_conv1d_81_layer_call_fn_799721
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
 ztrace_0

trace_02т
E__inference_conv1d_81_layer_call_and_return_conditional_losses_799737
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
 ztrace_0
:Fy 2kernel
:y 2bias
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
non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
/	variables
0trainable_variables
1regularization_losses
3__call__
*4&call_and_return_all_conditional_losses
&4"call_and_return_conditional_losses"
_generic_user_object
С
trace_0
trace_12
+__inference_dropout_83_layer_call_fn_799742
+__inference_dropout_83_layer_call_fn_799747Љ
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
 ztrace_0ztrace_1
ї
trace_0
trace_12М
F__inference_dropout_83_layer_call_and_return_conditional_losses_799759
F__inference_dropout_83_layer_call_and_return_conditional_losses_799764Љ
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
 ztrace_0ztrace_1
D
$_self_saveable_object_factories"
_generic_user_object
 "
trackable_dict_wrapper
.
=0
>1"
trackable_list_wrapper
.
=0
>1"
trackable_list_wrapper
 "
trackable_list_wrapper
В
non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
7	variables
8trainable_variables
9regularization_losses
;__call__
*<&call_and_return_all_conditional_losses
&<"call_and_return_conditional_losses"
_generic_user_object
х
trace_02Ц
)__inference_dense_89_layer_call_fn_799773
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
 ztrace_0

trace_02с
D__inference_dense_89_layer_call_and_return_conditional_losses_799804
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
:y 2kernel
: 2bias
 "
trackable_dict_wrapper
.
F0
G1"
trackable_list_wrapper
.
F0
G1"
trackable_list_wrapper
 "
trackable_list_wrapper
В
 non_trainable_variables
Ёlayers
Ђmetrics
 Ѓlayer_regularization_losses
Єlayer_metrics
@	variables
Atrainable_variables
Bregularization_losses
D__call__
*E&call_and_return_all_conditional_losses
&E"call_and_return_conditional_losses"
_generic_user_object
ц
Ѕtrace_02Ч
*__inference_conv1d_82_layer_call_fn_799813
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
 zЅtrace_0

Іtrace_02т
E__inference_conv1d_82_layer_call_and_return_conditional_losses_799837
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
 zІtrace_0
:ja 2kernel
:a 2bias
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
.
P0
Q1"
trackable_list_wrapper
.
P0
Q1"
trackable_list_wrapper
 "
trackable_list_wrapper
В
Їnon_trainable_variables
Јlayers
Љmetrics
 Њlayer_regularization_losses
Ћlayer_metrics
J	variables
Ktrainable_variables
Lregularization_losses
N__call__
*O&call_and_return_all_conditional_losses
&O"call_and_return_conditional_losses"
_generic_user_object
ц
Ќtrace_02Ч
*__inference_conv1d_83_layer_call_fn_799846
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
 zЌtrace_0

­trace_02т
E__inference_conv1d_83_layer_call_and_return_conditional_losses_799870
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
 z­trace_0
:sak 2kernel
:k 2bias
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
Ўnon_trainable_variables
Џlayers
Аmetrics
 Бlayer_regularization_losses
Вlayer_metrics
T	variables
Utrainable_variables
Vregularization_losses
X__call__
*Y&call_and_return_all_conditional_losses
&Y"call_and_return_conditional_losses"
_generic_user_object
ч
Гtrace_02Ш
+__inference_flatten_76_layer_call_fn_799875
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
 zГtrace_0

Дtrace_02у
F__inference_flatten_76_layer_call_and_return_conditional_losses_799881
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
 zДtrace_0
 "
trackable_dict_wrapper
 "
trackable_dict_wrapper
.
b0
c1"
trackable_list_wrapper
.
b0
c1"
trackable_list_wrapper
 "
trackable_list_wrapper
В
Еnon_trainable_variables
Жlayers
Зmetrics
 Иlayer_regularization_losses
Йlayer_metrics
\	variables
]trainable_variables
^regularization_losses
`__call__
*a&call_and_return_all_conditional_losses
&a"call_and_return_conditional_losses"
_generic_user_object
ь
Кtrace_02Э
0__inference_INJECTION_MASKS_layer_call_fn_799890
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
 zКtrace_0

Лtrace_02ш
K__inference_INJECTION_MASKS_layer_call_and_return_conditional_losses_799901
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
 zЛtrace_0
:k 2kernel
: 2bias
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
n
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
9
10"
trackable_list_wrapper
0
М0
Н1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
§Bњ
)__inference_model_76_layer_call_fn_799252	OFFSOURCEONSOURCE"Е
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
)__inference_model_76_layer_call_fn_799312	OFFSOURCEONSOURCE"Е
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
)__inference_model_76_layer_call_fn_799487inputs_offsourceinputs_onsource"Е
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
)__inference_model_76_layer_call_fn_799513inputs_offsourceinputs_onsource"Е
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
D__inference_model_76_layer_call_and_return_conditional_losses_799152	OFFSOURCEONSOURCE"Е
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
D__inference_model_76_layer_call_and_return_conditional_losses_799191	OFFSOURCEONSOURCE"Е
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
D__inference_model_76_layer_call_and_return_conditional_losses_799616inputs_offsourceinputs_onsource"Е
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
D__inference_model_76_layer_call_and_return_conditional_losses_799712inputs_offsourceinputs_onsource"Е
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
в
s0
О1
П2
Р3
С4
Т5
У6
Ф7
Х8
Ц9
Ч10
Ш11
Щ12
Ъ13
Ы14
Ь15
Э16
Ю17
Я18
а19
б20"
trackable_list_wrapper
:	  2	iteration
:  2learning_rate
 "
trackable_dict_wrapper
p
О0
Р1
Т2
Ф3
Ц4
Ш5
Ъ6
Ь7
Ю8
а9"
trackable_list_wrapper
p
П0
С1
У2
Х3
Ч4
Щ5
Ы6
Э7
Я8
б9"
trackable_list_wrapper
П
вtrace_0
гtrace_1
дtrace_2
еtrace_3
жtrace_4
зtrace_5
иtrace_6
йtrace_7
кtrace_8
лtrace_92Є
#__inference__update_step_xla_286496
#__inference__update_step_xla_286501
#__inference__update_step_xla_286506
#__inference__update_step_xla_286511
#__inference__update_step_xla_286516
#__inference__update_step_xla_286521
#__inference__update_step_xla_286526
#__inference__update_step_xla_286531
#__inference__update_step_xla_286536
#__inference__update_step_xla_286541Џ
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
 0zвtrace_0zгtrace_1zдtrace_2zеtrace_3zжtrace_4zзtrace_5zиtrace_6zйtrace_7zкtrace_8zлtrace_9
еBв
$__inference_signature_wrapper_799461	OFFSOURCEONSOURCE"
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
оBл
4__inference_whiten_passthrough_38_layer_call_fn_1190inputs"
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
љBі
O__inference_whiten_passthrough_38_layer_call_and_return_conditional_losses_1335inputs"
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
(__inference_reshape_76_layer_call_fn_733inputs"
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
C__inference_reshape_76_layer_call_and_return_conditional_losses_534inputs"
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
*__inference_conv1d_81_layer_call_fn_799721inputs"
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
E__inference_conv1d_81_layer_call_and_return_conditional_losses_799737inputs"
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
+__inference_dropout_83_layer_call_fn_799742inputs"Љ
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
+__inference_dropout_83_layer_call_fn_799747inputs"Љ
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
F__inference_dropout_83_layer_call_and_return_conditional_losses_799759inputs"Љ
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
F__inference_dropout_83_layer_call_and_return_conditional_losses_799764inputs"Љ
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
гBа
)__inference_dense_89_layer_call_fn_799773inputs"
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
юBы
D__inference_dense_89_layer_call_and_return_conditional_losses_799804inputs"
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
*__inference_conv1d_82_layer_call_fn_799813inputs"
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
E__inference_conv1d_82_layer_call_and_return_conditional_losses_799837inputs"
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
*__inference_conv1d_83_layer_call_fn_799846inputs"
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
E__inference_conv1d_83_layer_call_and_return_conditional_losses_799870inputs"
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
+__inference_flatten_76_layer_call_fn_799875inputs"
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
F__inference_flatten_76_layer_call_and_return_conditional_losses_799881inputs"
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
0__inference_INJECTION_MASKS_layer_call_fn_799890inputs"
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
K__inference_INJECTION_MASKS_layer_call_and_return_conditional_losses_799901inputs"
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
м	variables
н	keras_api

оtotal

пcount"
_tf_keras_metric
c
р	variables
с	keras_api

тtotal

уcount
ф
_fn_kwargs"
_tf_keras_metric
#:!Fy 2Adam/m/kernel
#:!Fy 2Adam/v/kernel
:y 2Adam/m/bias
:y 2Adam/v/bias
:y 2Adam/m/kernel
:y 2Adam/v/kernel
: 2Adam/m/bias
: 2Adam/v/bias
#:!ja 2Adam/m/kernel
#:!ja 2Adam/v/kernel
:a 2Adam/m/bias
:a 2Adam/v/bias
#:!sak 2Adam/m/kernel
#:!sak 2Adam/v/kernel
:k 2Adam/m/bias
:k 2Adam/v/bias
:k 2Adam/m/kernel
:k 2Adam/v/kernel
: 2Adam/m/bias
: 2Adam/v/bias
юBы
#__inference__update_step_xla_286496gradientvariable"­
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
#__inference__update_step_xla_286501gradientvariable"­
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
#__inference__update_step_xla_286506gradientvariable"­
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
#__inference__update_step_xla_286511gradientvariable"­
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
#__inference__update_step_xla_286516gradientvariable"­
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
#__inference__update_step_xla_286521gradientvariable"­
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
#__inference__update_step_xla_286526gradientvariable"­
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
#__inference__update_step_xla_286531gradientvariable"­
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
#__inference__update_step_xla_286536gradientvariable"­
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
#__inference__update_step_xla_286541gradientvariable"­
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
о0
п1"
trackable_list_wrapper
.
м	variables"
_generic_user_object
:  (2total
:  (2count
0
т0
у1"
trackable_list_wrapper
.
р	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapper
QbO
beta:0E__inference_conv1d_83_layer_call_and_return_conditional_losses_799870
TbR
	BiasAdd:0E__inference_conv1d_83_layer_call_and_return_conditional_losses_799870
QbO
beta:0E__inference_conv1d_83_layer_call_and_return_conditional_losses_799120
TbR
	BiasAdd:0E__inference_conv1d_83_layer_call_and_return_conditional_losses_799120
QbO
beta:0E__inference_conv1d_82_layer_call_and_return_conditional_losses_799837
TbR
	BiasAdd:0E__inference_conv1d_82_layer_call_and_return_conditional_losses_799837
QbO
beta:0E__inference_conv1d_82_layer_call_and_return_conditional_losses_799090
TbR
	BiasAdd:0E__inference_conv1d_82_layer_call_and_return_conditional_losses_799090
ZbX
conv1d_82/beta:0D__inference_model_76_layer_call_and_return_conditional_losses_799712
]b[
conv1d_82/BiasAdd:0D__inference_model_76_layer_call_and_return_conditional_losses_799712
ZbX
conv1d_83/beta:0D__inference_model_76_layer_call_and_return_conditional_losses_799712
]b[
conv1d_83/BiasAdd:0D__inference_model_76_layer_call_and_return_conditional_losses_799712
ZbX
conv1d_82/beta:0D__inference_model_76_layer_call_and_return_conditional_losses_799616
]b[
conv1d_82/BiasAdd:0D__inference_model_76_layer_call_and_return_conditional_losses_799616
ZbX
conv1d_83/beta:0D__inference_model_76_layer_call_and_return_conditional_losses_799616
]b[
conv1d_83/BiasAdd:0D__inference_model_76_layer_call_and_return_conditional_losses_799616
@b>
model_76/conv1d_82/beta:0!__inference__wrapped_model_798986
CbA
model_76/conv1d_82/BiasAdd:0!__inference__wrapped_model_798986
@b>
model_76/conv1d_83/beta:0!__inference__wrapped_model_798986
CbA
model_76/conv1d_83/BiasAdd:0!__inference__wrapped_model_798986В
K__inference_INJECTION_MASKS_layer_call_and_return_conditional_losses_799901cbc/Ђ,
%Ђ"
 
inputsџџџџџџџџџk
Њ ",Ђ)
"
tensor_0џџџџџџџџџ
 
0__inference_INJECTION_MASKS_layer_call_fn_799890Xbc/Ђ,
%Ђ"
 
inputsџџџџџџџџџk
Њ "!
unknownџџџџџџџџџ
#__inference__update_step_xla_286496vpЂm
fЂc

gradientFy
85	!Ђ
њFy

p
` VariableSpec 
`рЪПш?
Њ "
 
#__inference__update_step_xla_286501f`Ђ]
VЂS

gradienty
0-	Ђ
њy

p
` VariableSpec 
`рђш?
Њ "
 
#__inference__update_step_xla_286506nhЂe
^Ђ[

gradienty
41	Ђ
њy

p
` VariableSpec 
`рш?
Њ "
 
#__inference__update_step_xla_286511f`Ђ]
VЂS

gradient
0-	Ђ
њ

p
` VariableSpec 
`рюш?
Њ "
 
#__inference__update_step_xla_286516vpЂm
fЂc

gradientja
85	!Ђ
њja

p
` VariableSpec 
`рЇш?
Њ "
 
#__inference__update_step_xla_286521f`Ђ]
VЂS

gradienta
0-	Ђ
њa

p
` VariableSpec 
`рИЇш?
Њ "
 
#__inference__update_step_xla_286526vpЂm
fЂc

gradientsak
85	!Ђ
њsak

p
` VariableSpec 
`рЕЇш?
Њ "
 
#__inference__update_step_xla_286531f`Ђ]
VЂS

gradientk
0-	Ђ
њk

p
` VariableSpec 
`рУш?
Њ "
 
#__inference__update_step_xla_286536nhЂe
^Ђ[

gradientk
41	Ђ
њk

p
` VariableSpec 
`рАУш?
Њ "
 
#__inference__update_step_xla_286541f`Ђ]
VЂS

gradient
0-	Ђ
њ

p
` VariableSpec 
`ргУш?
Њ "
 і
!__inference__wrapped_model_798986а
+,=>FGPQbcЂ|
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
injection_masksџџџџџџџџџЕ
E__inference_conv1d_81_layer_call_and_return_conditional_losses_799737l+,4Ђ1
*Ђ'
%"
inputsџџџџџџџџџ
Њ "0Ђ-
&#
tensor_0џџџџџџџџџ2y
 
*__inference_conv1d_81_layer_call_fn_799721a+,4Ђ1
*Ђ'
%"
inputsџџџџџџџџџ
Њ "%"
unknownџџџџџџџџџ2yД
E__inference_conv1d_82_layer_call_and_return_conditional_losses_799837kFG3Ђ0
)Ђ&
$!
inputsџџџџџџџџџ2
Њ "0Ђ-
&#
tensor_0џџџџџџџџџa
 
*__inference_conv1d_82_layer_call_fn_799813`FG3Ђ0
)Ђ&
$!
inputsџџџџџџџџџ2
Њ "%"
unknownџџџџџџџџџaД
E__inference_conv1d_83_layer_call_and_return_conditional_losses_799870kPQ3Ђ0
)Ђ&
$!
inputsџџџџџџџџџa
Њ "0Ђ-
&#
tensor_0џџџџџџџџџk
 
*__inference_conv1d_83_layer_call_fn_799846`PQ3Ђ0
)Ђ&
$!
inputsџџџџџџџџџa
Њ "%"
unknownџџџџџџџџџkГ
D__inference_dense_89_layer_call_and_return_conditional_losses_799804k=>3Ђ0
)Ђ&
$!
inputsџџџџџџџџџ2y
Њ "0Ђ-
&#
tensor_0џџџџџџџџџ2
 
)__inference_dense_89_layer_call_fn_799773`=>3Ђ0
)Ђ&
$!
inputsџџџџџџџџџ2y
Њ "%"
unknownџџџџџџџџџ2Е
F__inference_dropout_83_layer_call_and_return_conditional_losses_799759k7Ђ4
-Ђ*
$!
inputsџџџџџџџџџ2y
p
Њ "0Ђ-
&#
tensor_0џџџџџџџџџ2y
 Е
F__inference_dropout_83_layer_call_and_return_conditional_losses_799764k7Ђ4
-Ђ*
$!
inputsџџџџџџџџџ2y
p 
Њ "0Ђ-
&#
tensor_0џџџџџџџџџ2y
 
+__inference_dropout_83_layer_call_fn_799742`7Ђ4
-Ђ*
$!
inputsџџџџџџџџџ2y
p
Њ "%"
unknownџџџџџџџџџ2y
+__inference_dropout_83_layer_call_fn_799747`7Ђ4
-Ђ*
$!
inputsџџџџџџџџџ2y
p 
Њ "%"
unknownџџџџџџџџџ2y­
F__inference_flatten_76_layer_call_and_return_conditional_losses_799881c3Ђ0
)Ђ&
$!
inputsџџџџџџџџџk
Њ ",Ђ)
"
tensor_0џџџџџџџџџk
 
+__inference_flatten_76_layer_call_fn_799875X3Ђ0
)Ђ&
$!
inputsџџџџџџџџџk
Њ "!
unknownџџџџџџџџџkќ
#__inference_internal_grad_fn_800005дхцЂ
|Ђy

 
,)
result_grads_0џџџџџџџџџk
,)
result_grads_1џџџџџџџџџk

result_grads_2 
Њ "B?

 
&#
tensor_1џџџџџџџџџk

tensor_2 ќ
#__inference_internal_grad_fn_800033дчшЂ
|Ђy

 
,)
result_grads_0џџџџџџџџџk
,)
result_grads_1џџџџџџџџџk

result_grads_2 
Њ "B?

 
&#
tensor_1џџџџџџџџџk

tensor_2 ќ
#__inference_internal_grad_fn_800061дщъЂ
|Ђy

 
,)
result_grads_0џџџџџџџџџa
,)
result_grads_1џџџџџџџџџa

result_grads_2 
Њ "B?

 
&#
tensor_1џџџџџџџџџa

tensor_2 ќ
#__inference_internal_grad_fn_800089дыьЂ
|Ђy

 
,)
result_grads_0џџџџџџџџџa
,)
result_grads_1џџџџџџџџџa

result_grads_2 
Њ "B?

 
&#
tensor_1џџџџџџџџџa

tensor_2 ќ
#__inference_internal_grad_fn_800117дэюЂ
|Ђy

 
,)
result_grads_0џџџџџџџџџa
,)
result_grads_1џџџџџџџџџa

result_grads_2 
Њ "B?

 
&#
tensor_1џџџџџџџџџa

tensor_2 ќ
#__inference_internal_grad_fn_800145дя№Ђ
|Ђy

 
,)
result_grads_0џџџџџџџџџk
,)
result_grads_1џџџџџџџџџk

result_grads_2 
Њ "B?

 
&#
tensor_1џџџџџџџџџk

tensor_2 ќ
#__inference_internal_grad_fn_800173дёђЂ
|Ђy

 
,)
result_grads_0џџџџџџџџџa
,)
result_grads_1џџџџџџџџџa

result_grads_2 
Њ "B?

 
&#
tensor_1џџџџџџџџџa

tensor_2 ќ
#__inference_internal_grad_fn_800201дѓєЂ
|Ђy

 
,)
result_grads_0џџџџџџџџџk
,)
result_grads_1џџџџџџџџџk

result_grads_2 
Њ "B?

 
&#
tensor_1џџџџџџџџџk

tensor_2 ќ
#__inference_internal_grad_fn_800229дѕіЂ
|Ђy

 
,)
result_grads_0џџџџџџџџџa
,)
result_grads_1џџџџџџџџџa

result_grads_2 
Њ "B?

 
&#
tensor_1џџџџџџџџџa

tensor_2 ќ
#__inference_internal_grad_fn_800257дїјЂ
|Ђy

 
,)
result_grads_0џџџџџџџџџk
,)
result_grads_1џџџџџџџџџk

result_grads_2 
Њ "B?

 
&#
tensor_1џџџџџџџџџk

tensor_2 
D__inference_model_76_layer_call_and_return_conditional_losses_799152Х
+,=>FGPQbcЂ
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
 
D__inference_model_76_layer_call_and_return_conditional_losses_799191Х
+,=>FGPQbcЂ
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
 
D__inference_model_76_layer_call_and_return_conditional_losses_799616е
+,=>FGPQbcЂ
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
 
D__inference_model_76_layer_call_and_return_conditional_losses_799712е
+,=>FGPQbcЂ
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
 ш
)__inference_model_76_layer_call_fn_799252К
+,=>FGPQbcЂ
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
unknownџџџџџџџџџш
)__inference_model_76_layer_call_fn_799312К
+,=>FGPQbcЂ
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
unknownџџџџџџџџџј
)__inference_model_76_layer_call_fn_799487Ъ
+,=>FGPQbcЂ
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
unknownџџџџџџџџџј
)__inference_model_76_layer_call_fn_799513Ъ
+,=>FGPQbcЂ
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
C__inference_reshape_76_layer_call_and_return_conditional_losses_534i4Ђ1
*Ђ'
%"
inputsџџџџџџџџџ
Њ "1Ђ.
'$
tensor_0џџџџџџџџџ
 
(__inference_reshape_76_layer_call_fn_733^4Ђ1
*Ђ'
%"
inputsџџџџџџџџџ
Њ "&#
unknownџџџџџџџџџє
$__inference_signature_wrapper_799461Ы
+,=>FGPQbczЂw
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
injection_masksџџџџџџџџџН
O__inference_whiten_passthrough_38_layer_call_and_return_conditional_losses_1335j5Ђ2
+Ђ(
&#
inputsџџџџџџџџџ
Њ "1Ђ.
'$
tensor_0џџџџџџџџџ
 
4__inference_whiten_passthrough_38_layer_call_fn_1190_5Ђ2
+Ђ(
&#
inputsџџџџџџџџџ
Њ "&#
unknownџџџџџџџџџ