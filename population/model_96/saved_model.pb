жШ
е
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
;
Elu
features"T
activations"T"
Ttype:
2
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
E
Relu
features"T
activations"T"
Ttype:
2	
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
9
Softmax
logits"T
softmax"T"
Ttype:
2
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
 "serve*2.12.12v2.12.0-25-g8e2b6655c0c8Щ
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
:f*
shared_nameAdam/v/kernel
o
!Adam/v/kernel/Read/ReadVariableOpReadVariableOpAdam/v/kernel*
_output_shapes

:f*
dtype0
v
Adam/m/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:f*
shared_nameAdam/m/kernel
o
!Adam/m/kernel/Read/ReadVariableOpReadVariableOpAdam/m/kernel*
_output_shapes

:f*
dtype0
r
Adam/v/bias_1VarHandleOp*
_output_shapes
: *
dtype0*
shape:f*
shared_nameAdam/v/bias_1
k
!Adam/v/bias_1/Read/ReadVariableOpReadVariableOpAdam/v/bias_1*
_output_shapes
:f*
dtype0
r
Adam/m/bias_1VarHandleOp*
_output_shapes
: *
dtype0*
shape:f*
shared_nameAdam/m/bias_1
k
!Adam/m/bias_1/Read/ReadVariableOpReadVariableOpAdam/m/bias_1*
_output_shapes
:f*
dtype0
~
Adam/v/kernel_1VarHandleOp*
_output_shapes
: *
dtype0*
shape:<|f* 
shared_nameAdam/v/kernel_1
w
#Adam/v/kernel_1/Read/ReadVariableOpReadVariableOpAdam/v/kernel_1*"
_output_shapes
:<|f*
dtype0
~
Adam/m/kernel_1VarHandleOp*
_output_shapes
: *
dtype0*
shape:<|f* 
shared_nameAdam/m/kernel_1
w
#Adam/m/kernel_1/Read/ReadVariableOpReadVariableOpAdam/m/kernel_1*"
_output_shapes
:<|f*
dtype0
r
Adam/v/bias_2VarHandleOp*
_output_shapes
: *
dtype0*
shape:|*
shared_nameAdam/v/bias_2
k
!Adam/v/bias_2/Read/ReadVariableOpReadVariableOpAdam/v/bias_2*
_output_shapes
:|*
dtype0
r
Adam/m/bias_2VarHandleOp*
_output_shapes
: *
dtype0*
shape:|*
shared_nameAdam/m/bias_2
k
!Adam/m/bias_2/Read/ReadVariableOpReadVariableOpAdam/m/bias_2*
_output_shapes
:|*
dtype0
z
Adam/v/kernel_2VarHandleOp*
_output_shapes
: *
dtype0*
shape
:X|* 
shared_nameAdam/v/kernel_2
s
#Adam/v/kernel_2/Read/ReadVariableOpReadVariableOpAdam/v/kernel_2*
_output_shapes

:X|*
dtype0
z
Adam/m/kernel_2VarHandleOp*
_output_shapes
: *
dtype0*
shape
:X|* 
shared_nameAdam/m/kernel_2
s
#Adam/m/kernel_2/Read/ReadVariableOpReadVariableOpAdam/m/kernel_2*
_output_shapes

:X|*
dtype0
r
Adam/v/bias_3VarHandleOp*
_output_shapes
: *
dtype0*
shape:X*
shared_nameAdam/v/bias_3
k
!Adam/v/bias_3/Read/ReadVariableOpReadVariableOpAdam/v/bias_3*
_output_shapes
:X*
dtype0
r
Adam/m/bias_3VarHandleOp*
_output_shapes
: *
dtype0*
shape:X*
shared_nameAdam/m/bias_3
k
!Adam/m/bias_3/Read/ReadVariableOpReadVariableOpAdam/m/bias_3*
_output_shapes
:X*
dtype0
z
Adam/v/kernel_3VarHandleOp*
_output_shapes
: *
dtype0*
shape
:tX* 
shared_nameAdam/v/kernel_3
s
#Adam/v/kernel_3/Read/ReadVariableOpReadVariableOpAdam/v/kernel_3*
_output_shapes

:tX*
dtype0
z
Adam/m/kernel_3VarHandleOp*
_output_shapes
: *
dtype0*
shape
:tX* 
shared_nameAdam/m/kernel_3
s
#Adam/m/kernel_3/Read/ReadVariableOpReadVariableOpAdam/m/kernel_3*
_output_shapes

:tX*
dtype0
r
Adam/v/bias_4VarHandleOp*
_output_shapes
: *
dtype0*
shape:t*
shared_nameAdam/v/bias_4
k
!Adam/v/bias_4/Read/ReadVariableOpReadVariableOpAdam/v/bias_4*
_output_shapes
:t*
dtype0
r
Adam/m/bias_4VarHandleOp*
_output_shapes
: *
dtype0*
shape:t*
shared_nameAdam/m/bias_4
k
!Adam/m/bias_4/Read/ReadVariableOpReadVariableOpAdam/m/bias_4*
_output_shapes
:t*
dtype0
~
Adam/v/kernel_4VarHandleOp*
_output_shapes
: *
dtype0*
shape:[t* 
shared_nameAdam/v/kernel_4
w
#Adam/v/kernel_4/Read/ReadVariableOpReadVariableOpAdam/v/kernel_4*"
_output_shapes
:[t*
dtype0
~
Adam/m/kernel_4VarHandleOp*
_output_shapes
: *
dtype0*
shape:[t* 
shared_nameAdam/m/kernel_4
w
#Adam/m/kernel_4/Read/ReadVariableOpReadVariableOpAdam/m/kernel_4*"
_output_shapes
:[t*
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
:f*
shared_namekernel
a
kernel/Read/ReadVariableOpReadVariableOpkernel*
_output_shapes

:f*
dtype0
d
bias_1VarHandleOp*
_output_shapes
: *
dtype0*
shape:f*
shared_namebias_1
]
bias_1/Read/ReadVariableOpReadVariableOpbias_1*
_output_shapes
:f*
dtype0
p
kernel_1VarHandleOp*
_output_shapes
: *
dtype0*
shape:<|f*
shared_name
kernel_1
i
kernel_1/Read/ReadVariableOpReadVariableOpkernel_1*"
_output_shapes
:<|f*
dtype0
d
bias_2VarHandleOp*
_output_shapes
: *
dtype0*
shape:|*
shared_namebias_2
]
bias_2/Read/ReadVariableOpReadVariableOpbias_2*
_output_shapes
:|*
dtype0
l
kernel_2VarHandleOp*
_output_shapes
: *
dtype0*
shape
:X|*
shared_name
kernel_2
e
kernel_2/Read/ReadVariableOpReadVariableOpkernel_2*
_output_shapes

:X|*
dtype0
d
bias_3VarHandleOp*
_output_shapes
: *
dtype0*
shape:X*
shared_namebias_3
]
bias_3/Read/ReadVariableOpReadVariableOpbias_3*
_output_shapes
:X*
dtype0
l
kernel_3VarHandleOp*
_output_shapes
: *
dtype0*
shape
:tX*
shared_name
kernel_3
e
kernel_3/Read/ReadVariableOpReadVariableOpkernel_3*
_output_shapes

:tX*
dtype0
d
bias_4VarHandleOp*
_output_shapes
: *
dtype0*
shape:t*
shared_namebias_4
]
bias_4/Read/ReadVariableOpReadVariableOpbias_4*
_output_shapes
:t*
dtype0
p
kernel_4VarHandleOp*
_output_shapes
: *
dtype0*
shape:[t*
shared_name
kernel_4
i
kernel_4/Read/ReadVariableOpReadVariableOpkernel_4*"
_output_shapes
:[t*
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
$__inference_signature_wrapper_542628

NoOpNoOp
гT
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*T
valueTBT BњS

layer-0
layer-1
layer-2
layer_with_weights-0
layer-3
layer_with_weights-1
layer-4
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
Ы
/	variables
0trainable_variables
1regularization_losses
2	keras_api
3__call__
*4&call_and_return_all_conditional_losses

5kernel
6bias
#7_self_saveable_object_factories*
Г
8	variables
9trainable_variables
:regularization_losses
;	keras_api
<__call__
*=&call_and_return_all_conditional_losses
#>_self_saveable_object_factories* 
Ы
?	variables
@trainable_variables
Aregularization_losses
B	keras_api
C__call__
*D&call_and_return_all_conditional_losses

Ekernel
Fbias
#G_self_saveable_object_factories*
э
H	variables
Itrainable_variables
Jregularization_losses
K	keras_api
L__call__
*M&call_and_return_all_conditional_losses

Nkernel
Obias
#P_self_saveable_object_factories
 Q_jit_compiled_convolution_op*
Г
R	variables
Strainable_variables
Tregularization_losses
U	keras_api
V__call__
*W&call_and_return_all_conditional_losses
#X_self_saveable_object_factories* 
'
#Y_self_saveable_object_factories* 
Ы
Z	variables
[trainable_variables
\regularization_losses
]	keras_api
^__call__
*_&call_and_return_all_conditional_losses

`kernel
abias
#b_self_saveable_object_factories*
J
+0
,1
52
63
E4
F5
N6
O7
`8
a9*
J
+0
,1
52
63
E4
F5
N6
O7
`8
a9*
* 
А
cnon_trainable_variables

dlayers
emetrics
flayer_regularization_losses
glayer_metrics
	variables
trainable_variables
regularization_losses
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*
6
htrace_0
itrace_1
jtrace_2
ktrace_3* 
6
ltrace_0
mtrace_1
ntrace_2
otrace_3* 
* 

p
_variables
q_iterations
r_learning_rate
s_index_dict
t
_momentums
u_velocities
v_update_step_xla*

wserving_default* 
* 
* 
* 
* 
* 

xnon_trainable_variables

ylayers
zmetrics
{layer_regularization_losses
|layer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses* 

}trace_0* 

~trace_0* 
* 
* 
* 
* 

non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
	variables
trainable_variables
 regularization_losses
"__call__
*#&call_and_return_all_conditional_losses
&#"call_and_return_conditional_losses* 

trace_0* 

trace_0* 
* 

+0
,1*

+0
,1*
* 

non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
%	variables
&trainable_variables
'regularization_losses
)__call__
**&call_and_return_all_conditional_losses
&*"call_and_return_conditional_losses*

trace_0* 

trace_0* 
XR
VARIABLE_VALUEkernel_46layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE*
TN
VARIABLE_VALUEbias_44layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 

50
61*

50
61*
* 

non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
/	variables
0trainable_variables
1regularization_losses
3__call__
*4&call_and_return_all_conditional_losses
&4"call_and_return_conditional_losses*

trace_0* 

trace_0* 
XR
VARIABLE_VALUEkernel_36layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE*
TN
VARIABLE_VALUEbias_34layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
* 

non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
8	variables
9trainable_variables
:regularization_losses
<__call__
*=&call_and_return_all_conditional_losses
&="call_and_return_conditional_losses* 

trace_0* 

trace_0* 
* 

E0
F1*

E0
F1*
* 

non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
?	variables
@trainable_variables
Aregularization_losses
C__call__
*D&call_and_return_all_conditional_losses
&D"call_and_return_conditional_losses*

 trace_0* 

Ёtrace_0* 
XR
VARIABLE_VALUEkernel_26layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE*
TN
VARIABLE_VALUEbias_24layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 

N0
O1*

N0
O1*
* 

Ђnon_trainable_variables
Ѓlayers
Єmetrics
 Ѕlayer_regularization_losses
Іlayer_metrics
H	variables
Itrainable_variables
Jregularization_losses
L__call__
*M&call_and_return_all_conditional_losses
&M"call_and_return_conditional_losses*

Їtrace_0* 

Јtrace_0* 
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
Љnon_trainable_variables
Њlayers
Ћmetrics
 Ќlayer_regularization_losses
­layer_metrics
R	variables
Strainable_variables
Tregularization_losses
V__call__
*W&call_and_return_all_conditional_losses
&W"call_and_return_conditional_losses* 

Ўtrace_0* 

Џtrace_0* 
* 
* 

`0
a1*

`0
a1*
* 

Аnon_trainable_variables
Бlayers
Вmetrics
 Гlayer_regularization_losses
Дlayer_metrics
Z	variables
[trainable_variables
\regularization_losses
^__call__
*_&call_and_return_all_conditional_losses
&_"call_and_return_conditional_losses*

Еtrace_0* 

Жtrace_0* 
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
З0
И1*
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
q0
Й1
К2
Л3
М4
Н5
О6
П7
Р8
С9
Т10
У11
Ф12
Х13
Ц14
Ч15
Ш16
Щ17
Ъ18
Ы19
Ь20*
SM
VARIABLE_VALUE	iteration0optimizer/_iterations/.ATTRIBUTES/VARIABLE_VALUE*
ZT
VARIABLE_VALUElearning_rate3optimizer/_learning_rate/.ATTRIBUTES/VARIABLE_VALUE*
* 
T
Й0
Л1
Н2
П3
С4
У5
Х6
Ч7
Щ8
Ы9*
T
К0
М1
О2
Р3
Т4
Ф5
Ц6
Ш7
Ъ8
Ь9*

Эtrace_0
Юtrace_1
Яtrace_2
аtrace_3
бtrace_4
вtrace_5
гtrace_6
дtrace_7
еtrace_8
жtrace_9* 
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
з	variables
и	keras_api

йtotal

кcount*
M
л	variables
м	keras_api

нtotal

оcount
п
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
й0
к1*

з	variables*
UO
VARIABLE_VALUEtotal_14keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE*
UO
VARIABLE_VALUEcount_14keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE*

н0
о1*

л	variables*
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
__inference__traced_save_543454
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
"__inference__traced_restore_543572Ву
п

0__inference_INJECTION_MASKS_layer_call_fn_543063

inputs
unknown:f
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
K__inference_INJECTION_MASKS_layer_call_and_return_conditional_losses_542328o
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
:џџџџџџџџџf: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:џџџџџџџџџf
 
_user_specified_nameinputs
Ў

"__inference__traced_restore_543572
file_prefix/
assignvariableop_kernel_4:[t'
assignvariableop_1_bias_4:t-
assignvariableop_2_kernel_3:tX'
assignvariableop_3_bias_3:X-
assignvariableop_4_kernel_2:X|'
assignvariableop_5_bias_2:|1
assignvariableop_6_kernel_1:<|f'
assignvariableop_7_bias_1:f+
assignvariableop_8_kernel:f%
assignvariableop_9_bias:'
assignvariableop_10_iteration:	 +
!assignvariableop_11_learning_rate: 9
#assignvariableop_12_adam_m_kernel_4:[t9
#assignvariableop_13_adam_v_kernel_4:[t/
!assignvariableop_14_adam_m_bias_4:t/
!assignvariableop_15_adam_v_bias_4:t5
#assignvariableop_16_adam_m_kernel_3:tX5
#assignvariableop_17_adam_v_kernel_3:tX/
!assignvariableop_18_adam_m_bias_3:X/
!assignvariableop_19_adam_v_bias_3:X5
#assignvariableop_20_adam_m_kernel_2:X|5
#assignvariableop_21_adam_v_kernel_2:X|/
!assignvariableop_22_adam_m_bias_2:|/
!assignvariableop_23_adam_v_bias_2:|9
#assignvariableop_24_adam_m_kernel_1:<|f9
#assignvariableop_25_adam_v_kernel_1:<|f/
!assignvariableop_26_adam_m_bias_1:f/
!assignvariableop_27_adam_v_bias_1:f3
!assignvariableop_28_adam_m_kernel:f3
!assignvariableop_29_adam_v_kernel:f-
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
И
O
#__inference__update_step_xla_285994
gradient
variable:X|*
_XlaMustCompile(*(
_construction_contextkEagerRuntime*
_input_shapes
:X|: *
	_noinline(:($
"
_user_specified_name
variable:H D

_output_shapes

:X|
"
_user_specified_name
gradient
Х

E__inference_conv1d_96_layer_call_and_return_conditional_losses_542303

inputsA
+conv1d_expanddims_1_readvariableop_resource:<|f-
biasadd_readvariableop_resource:f
identityЂBiasAdd/ReadVariableOpЂ"Conv1D/ExpandDims_1/ReadVariableOp`
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
:џџџџџџџџџ|
"Conv1D/ExpandDims_1/ReadVariableOpReadVariableOp+conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:<|f*
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
:<|fЌ
Conv1DConv2DConv1D/ExpandDims:output:0Conv1D/ExpandDims_1:output:0*
T0*/
_output_shapes
:џџџџџџџџџf*
paddingSAME*
strides
B
Conv1D/SqueezeSqueezeConv1D:output:0*
T0*+
_output_shapes
:џџџџџџџџџf*
squeeze_dims

§џџџџџџџџr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:f*
dtype0
BiasAddBiasAddConv1D/Squeeze:output:0BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:џџџџџџџџџfR
EluEluBiasAdd:output:0*
T0*+
_output_shapes
:џџџџџџџџџfd
IdentityIdentityElu:activations:0^NoOp*
T0*+
_output_shapes
:џџџџџџџџџf
NoOpNoOp^BiasAdd/ReadVariableOp#^Conv1D/ExpandDims_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:џџџџџџџџџ|: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2H
"Conv1D/ExpandDims_1/ReadVariableOp"Conv1D/ExpandDims_1/ReadVariableOp:S O
+
_output_shapes
:џџџџџџџџџ|
 
_user_specified_nameinputs
И
G
+__inference_flatten_96_layer_call_fn_543048

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
:џџџџџџџџџf* 
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
F__inference_flatten_96_layer_call_and_return_conditional_losses_542315`
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:џџџџџџџџџf"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:џџџџџџџџџf:S O
+
_output_shapes
:џџџџџџџџџf
 
_user_specified_nameinputs
п
`
D__inference_reshape_96_layer_call_and_return_conditional_losses_1444

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
є
j
N__inference_whiten_passthrough_50_layer_call_and_return_conditional_losses_833

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
__inference_crop_samples_816I
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
б
i
M__inference_max_pooling1d_115_layer_call_and_return_conditional_losses_542978

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
*
paddingSAME*
strides
	
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



!__inference__wrapped_model_542160
	offsource
onsourceT
>model_96_conv1d_95_conv1d_expanddims_1_readvariableop_resource:[t@
2model_96_conv1d_95_biasadd_readvariableop_resource:tF
4model_96_dense_116_tensordot_readvariableop_resource:tX@
2model_96_dense_116_biasadd_readvariableop_resource:XF
4model_96_dense_117_tensordot_readvariableop_resource:X|@
2model_96_dense_117_biasadd_readvariableop_resource:|T
>model_96_conv1d_96_conv1d_expanddims_1_readvariableop_resource:<|f@
2model_96_conv1d_96_biasadd_readvariableop_resource:fI
7model_96_injection_masks_matmul_readvariableop_resource:fF
8model_96_injection_masks_biasadd_readvariableop_resource:
identityЂ/model_96/INJECTION_MASKS/BiasAdd/ReadVariableOpЂ.model_96/INJECTION_MASKS/MatMul/ReadVariableOpЂ)model_96/conv1d_95/BiasAdd/ReadVariableOpЂ5model_96/conv1d_95/Conv1D/ExpandDims_1/ReadVariableOpЂ)model_96/conv1d_96/BiasAdd/ReadVariableOpЂ5model_96/conv1d_96/Conv1D/ExpandDims_1/ReadVariableOpЂ)model_96/dense_116/BiasAdd/ReadVariableOpЂ+model_96/dense_116/Tensordot/ReadVariableOpЂ)model_96/dense_117/BiasAdd/ReadVariableOpЂ+model_96/dense_117/Tensordot/ReadVariableOpЪ
.model_96/whiten_passthrough_50/PartitionedCallPartitionedCall	offsource*
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
)__inference_restored_function_body_285129э
#model_96/reshape_96/PartitionedCallPartitionedCall7model_96/whiten_passthrough_50/PartitionedCall:output:0*
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
)__inference_restored_function_body_285135s
(model_96/conv1d_95/Conv1D/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
§џџџџџџџџЮ
$model_96/conv1d_95/Conv1D/ExpandDims
ExpandDims,model_96/reshape_96/PartitionedCall:output:01model_96/conv1d_95/Conv1D/ExpandDims/dim:output:0*
T0*0
_output_shapes
:џџџџџџџџџИ
5model_96/conv1d_95/Conv1D/ExpandDims_1/ReadVariableOpReadVariableOp>model_96_conv1d_95_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:[t*
dtype0l
*model_96/conv1d_95/Conv1D/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : й
&model_96/conv1d_95/Conv1D/ExpandDims_1
ExpandDims=model_96/conv1d_95/Conv1D/ExpandDims_1/ReadVariableOp:value:03model_96/conv1d_95/Conv1D/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:[tх
model_96/conv1d_95/Conv1DConv2D-model_96/conv1d_95/Conv1D/ExpandDims:output:0/model_96/conv1d_95/Conv1D/ExpandDims_1:output:0*
T0*/
_output_shapes
:џџџџџџџџџt*
paddingSAME*
strides
mІ
!model_96/conv1d_95/Conv1D/SqueezeSqueeze"model_96/conv1d_95/Conv1D:output:0*
T0*+
_output_shapes
:џџџџџџџџџt*
squeeze_dims

§џџџџџџџџ
)model_96/conv1d_95/BiasAdd/ReadVariableOpReadVariableOp2model_96_conv1d_95_biasadd_readvariableop_resource*
_output_shapes
:t*
dtype0К
model_96/conv1d_95/BiasAddBiasAdd*model_96/conv1d_95/Conv1D/Squeeze:output:01model_96/conv1d_95/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:џџџџџџџџџt
model_96/conv1d_95/SoftmaxSoftmax#model_96/conv1d_95/BiasAdd:output:0*
T0*+
_output_shapes
:џџџџџџџџџt 
+model_96/dense_116/Tensordot/ReadVariableOpReadVariableOp4model_96_dense_116_tensordot_readvariableop_resource*
_output_shapes

:tX*
dtype0k
!model_96/dense_116/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:r
!model_96/dense_116/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       
"model_96/dense_116/Tensordot/ShapeShape$model_96/conv1d_95/Softmax:softmax:0*
T0*
_output_shapes
::эЯl
*model_96/dense_116/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 
%model_96/dense_116/Tensordot/GatherV2GatherV2+model_96/dense_116/Tensordot/Shape:output:0*model_96/dense_116/Tensordot/free:output:03model_96/dense_116/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:n
,model_96/dense_116/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 
'model_96/dense_116/Tensordot/GatherV2_1GatherV2+model_96/dense_116/Tensordot/Shape:output:0*model_96/dense_116/Tensordot/axes:output:05model_96/dense_116/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:l
"model_96/dense_116/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: Ї
!model_96/dense_116/Tensordot/ProdProd.model_96/dense_116/Tensordot/GatherV2:output:0+model_96/dense_116/Tensordot/Const:output:0*
T0*
_output_shapes
: n
$model_96/dense_116/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: ­
#model_96/dense_116/Tensordot/Prod_1Prod0model_96/dense_116/Tensordot/GatherV2_1:output:0-model_96/dense_116/Tensordot/Const_1:output:0*
T0*
_output_shapes
: j
(model_96/dense_116/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ш
#model_96/dense_116/Tensordot/concatConcatV2*model_96/dense_116/Tensordot/free:output:0*model_96/dense_116/Tensordot/axes:output:01model_96/dense_116/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:В
"model_96/dense_116/Tensordot/stackPack*model_96/dense_116/Tensordot/Prod:output:0,model_96/dense_116/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:Н
&model_96/dense_116/Tensordot/transpose	Transpose$model_96/conv1d_95/Softmax:softmax:0,model_96/dense_116/Tensordot/concat:output:0*
T0*+
_output_shapes
:џџџџџџџџџtУ
$model_96/dense_116/Tensordot/ReshapeReshape*model_96/dense_116/Tensordot/transpose:y:0+model_96/dense_116/Tensordot/stack:output:0*
T0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџУ
#model_96/dense_116/Tensordot/MatMulMatMul-model_96/dense_116/Tensordot/Reshape:output:03model_96/dense_116/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџXn
$model_96/dense_116/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:Xl
*model_96/dense_116/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ѓ
%model_96/dense_116/Tensordot/concat_1ConcatV2.model_96/dense_116/Tensordot/GatherV2:output:0-model_96/dense_116/Tensordot/Const_2:output:03model_96/dense_116/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:М
model_96/dense_116/TensordotReshape-model_96/dense_116/Tensordot/MatMul:product:0.model_96/dense_116/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:џџџџџџџџџX
)model_96/dense_116/BiasAdd/ReadVariableOpReadVariableOp2model_96_dense_116_biasadd_readvariableop_resource*
_output_shapes
:X*
dtype0Е
model_96/dense_116/BiasAddBiasAdd%model_96/dense_116/Tensordot:output:01model_96/dense_116/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:џџџџџџџџџX\
model_96/dense_116/betaConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
model_96/dense_116/mulMul model_96/dense_116/beta:output:0#model_96/dense_116/BiasAdd:output:0*
T0*+
_output_shapes
:џџџџџџџџџXw
model_96/dense_116/SigmoidSigmoidmodel_96/dense_116/mul:z:0*
T0*+
_output_shapes
:џџџџџџџџџX
model_96/dense_116/mul_1Mul#model_96/dense_116/BiasAdd:output:0model_96/dense_116/Sigmoid:y:0*
T0*+
_output_shapes
:џџџџџџџџџX{
model_96/dense_116/IdentityIdentitymodel_96/dense_116/mul_1:z:0*
T0*+
_output_shapes
:џџџџџџџџџX
model_96/dense_116/IdentityN	IdentityNmodel_96/dense_116/mul_1:z:0#model_96/dense_116/BiasAdd:output:0 model_96/dense_116/beta:output:0*
T
2*,
_gradient_op_typeCustomGradient-542099*D
_output_shapes2
0:џџџџџџџџџX:џџџџџџџџџX: k
)model_96/max_pooling1d_115/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :Ш
%model_96/max_pooling1d_115/ExpandDims
ExpandDims%model_96/dense_116/IdentityN:output:02model_96/max_pooling1d_115/ExpandDims/dim:output:0*
T0*/
_output_shapes
:џџџџџџџџџXЩ
"model_96/max_pooling1d_115/MaxPoolMaxPool.model_96/max_pooling1d_115/ExpandDims:output:0*/
_output_shapes
:џџџџџџџџџX*
ksize
*
paddingSAME*
strides
	Ї
"model_96/max_pooling1d_115/SqueezeSqueeze+model_96/max_pooling1d_115/MaxPool:output:0*
T0*+
_output_shapes
:џџџџџџџџџX*
squeeze_dims
 
+model_96/dense_117/Tensordot/ReadVariableOpReadVariableOp4model_96_dense_117_tensordot_readvariableop_resource*
_output_shapes

:X|*
dtype0k
!model_96/dense_117/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:r
!model_96/dense_117/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       
"model_96/dense_117/Tensordot/ShapeShape+model_96/max_pooling1d_115/Squeeze:output:0*
T0*
_output_shapes
::эЯl
*model_96/dense_117/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 
%model_96/dense_117/Tensordot/GatherV2GatherV2+model_96/dense_117/Tensordot/Shape:output:0*model_96/dense_117/Tensordot/free:output:03model_96/dense_117/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:n
,model_96/dense_117/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 
'model_96/dense_117/Tensordot/GatherV2_1GatherV2+model_96/dense_117/Tensordot/Shape:output:0*model_96/dense_117/Tensordot/axes:output:05model_96/dense_117/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:l
"model_96/dense_117/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: Ї
!model_96/dense_117/Tensordot/ProdProd.model_96/dense_117/Tensordot/GatherV2:output:0+model_96/dense_117/Tensordot/Const:output:0*
T0*
_output_shapes
: n
$model_96/dense_117/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: ­
#model_96/dense_117/Tensordot/Prod_1Prod0model_96/dense_117/Tensordot/GatherV2_1:output:0-model_96/dense_117/Tensordot/Const_1:output:0*
T0*
_output_shapes
: j
(model_96/dense_117/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ш
#model_96/dense_117/Tensordot/concatConcatV2*model_96/dense_117/Tensordot/free:output:0*model_96/dense_117/Tensordot/axes:output:01model_96/dense_117/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:В
"model_96/dense_117/Tensordot/stackPack*model_96/dense_117/Tensordot/Prod:output:0,model_96/dense_117/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:Ф
&model_96/dense_117/Tensordot/transpose	Transpose+model_96/max_pooling1d_115/Squeeze:output:0,model_96/dense_117/Tensordot/concat:output:0*
T0*+
_output_shapes
:џџџџџџџџџXУ
$model_96/dense_117/Tensordot/ReshapeReshape*model_96/dense_117/Tensordot/transpose:y:0+model_96/dense_117/Tensordot/stack:output:0*
T0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџУ
#model_96/dense_117/Tensordot/MatMulMatMul-model_96/dense_117/Tensordot/Reshape:output:03model_96/dense_117/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ|n
$model_96/dense_117/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:|l
*model_96/dense_117/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ѓ
%model_96/dense_117/Tensordot/concat_1ConcatV2.model_96/dense_117/Tensordot/GatherV2:output:0-model_96/dense_117/Tensordot/Const_2:output:03model_96/dense_117/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:М
model_96/dense_117/TensordotReshape-model_96/dense_117/Tensordot/MatMul:product:0.model_96/dense_117/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:џџџџџџџџџ|
)model_96/dense_117/BiasAdd/ReadVariableOpReadVariableOp2model_96_dense_117_biasadd_readvariableop_resource*
_output_shapes
:|*
dtype0Е
model_96/dense_117/BiasAddBiasAdd%model_96/dense_117/Tensordot:output:01model_96/dense_117/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:џџџџџџџџџ|z
model_96/dense_117/ReluRelu#model_96/dense_117/BiasAdd:output:0*
T0*+
_output_shapes
:џџџџџџџџџ|s
(model_96/conv1d_96/Conv1D/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
§џџџџџџџџЦ
$model_96/conv1d_96/Conv1D/ExpandDims
ExpandDims%model_96/dense_117/Relu:activations:01model_96/conv1d_96/Conv1D/ExpandDims/dim:output:0*
T0*/
_output_shapes
:џџџџџџџџџ|И
5model_96/conv1d_96/Conv1D/ExpandDims_1/ReadVariableOpReadVariableOp>model_96_conv1d_96_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:<|f*
dtype0l
*model_96/conv1d_96/Conv1D/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : й
&model_96/conv1d_96/Conv1D/ExpandDims_1
ExpandDims=model_96/conv1d_96/Conv1D/ExpandDims_1/ReadVariableOp:value:03model_96/conv1d_96/Conv1D/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:<|fх
model_96/conv1d_96/Conv1DConv2D-model_96/conv1d_96/Conv1D/ExpandDims:output:0/model_96/conv1d_96/Conv1D/ExpandDims_1:output:0*
T0*/
_output_shapes
:џџџџџџџџџf*
paddingSAME*
strides
BІ
!model_96/conv1d_96/Conv1D/SqueezeSqueeze"model_96/conv1d_96/Conv1D:output:0*
T0*+
_output_shapes
:џџџџџџџџџf*
squeeze_dims

§џџџџџџџџ
)model_96/conv1d_96/BiasAdd/ReadVariableOpReadVariableOp2model_96_conv1d_96_biasadd_readvariableop_resource*
_output_shapes
:f*
dtype0К
model_96/conv1d_96/BiasAddBiasAdd*model_96/conv1d_96/Conv1D/Squeeze:output:01model_96/conv1d_96/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:џџџџџџџџџfx
model_96/conv1d_96/EluElu#model_96/conv1d_96/BiasAdd:output:0*
T0*+
_output_shapes
:џџџџџџџџџfj
model_96/flatten_96/ConstConst*
_output_shapes
:*
dtype0*
valueB"џџџџf   Ђ
model_96/flatten_96/ReshapeReshape$model_96/conv1d_96/Elu:activations:0"model_96/flatten_96/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџfІ
.model_96/INJECTION_MASKS/MatMul/ReadVariableOpReadVariableOp7model_96_injection_masks_matmul_readvariableop_resource*
_output_shapes

:f*
dtype0Й
model_96/INJECTION_MASKS/MatMulMatMul$model_96/flatten_96/Reshape:output:06model_96/INJECTION_MASKS/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџЄ
/model_96/INJECTION_MASKS/BiasAdd/ReadVariableOpReadVariableOp8model_96_injection_masks_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0С
 model_96/INJECTION_MASKS/BiasAddBiasAdd)model_96/INJECTION_MASKS/MatMul:product:07model_96/INJECTION_MASKS/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ
 model_96/INJECTION_MASKS/SigmoidSigmoid)model_96/INJECTION_MASKS/BiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџs
IdentityIdentity$model_96/INJECTION_MASKS/Sigmoid:y:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџЅ
NoOpNoOp0^model_96/INJECTION_MASKS/BiasAdd/ReadVariableOp/^model_96/INJECTION_MASKS/MatMul/ReadVariableOp*^model_96/conv1d_95/BiasAdd/ReadVariableOp6^model_96/conv1d_95/Conv1D/ExpandDims_1/ReadVariableOp*^model_96/conv1d_96/BiasAdd/ReadVariableOp6^model_96/conv1d_96/Conv1D/ExpandDims_1/ReadVariableOp*^model_96/dense_116/BiasAdd/ReadVariableOp,^model_96/dense_116/Tensordot/ReadVariableOp*^model_96/dense_117/BiasAdd/ReadVariableOp,^model_96/dense_117/Tensordot/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*X
_input_shapesG
E:џџџџџџџџџ:џџџџџџџџџ : : : : : : : : : : 2b
/model_96/INJECTION_MASKS/BiasAdd/ReadVariableOp/model_96/INJECTION_MASKS/BiasAdd/ReadVariableOp2`
.model_96/INJECTION_MASKS/MatMul/ReadVariableOp.model_96/INJECTION_MASKS/MatMul/ReadVariableOp2V
)model_96/conv1d_95/BiasAdd/ReadVariableOp)model_96/conv1d_95/BiasAdd/ReadVariableOp2n
5model_96/conv1d_95/Conv1D/ExpandDims_1/ReadVariableOp5model_96/conv1d_95/Conv1D/ExpandDims_1/ReadVariableOp2V
)model_96/conv1d_96/BiasAdd/ReadVariableOp)model_96/conv1d_96/BiasAdd/ReadVariableOp2n
5model_96/conv1d_96/Conv1D/ExpandDims_1/ReadVariableOp5model_96/conv1d_96/Conv1D/ExpandDims_1/ReadVariableOp2V
)model_96/dense_116/BiasAdd/ReadVariableOp)model_96/dense_116/BiasAdd/ReadVariableOp2Z
+model_96/dense_116/Tensordot/ReadVariableOp+model_96/dense_116/Tensordot/ReadVariableOp2V
)model_96/dense_117/BiasAdd/ReadVariableOp)model_96/dense_117/BiasAdd/ReadVariableOp2Z
+model_96/dense_117/Tensordot/ReadVariableOp+model_96/dense_117/Tensordot/ReadVariableOp:VR
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
И
O
#__inference__update_step_xla_286014
gradient
variable:f*
_XlaMustCompile(*(
_construction_contextkEagerRuntime*
_input_shapes
:f: *
	_noinline(:($
"
_user_specified_name
variable:H D

_output_shapes

:f
"
_user_specified_name
gradient
Ќ
K
#__inference__update_step_xla_286009
gradient
variable:f*
_XlaMustCompile(*(
_construction_contextkEagerRuntime*
_input_shapes

:f: *
	_noinline(:($
"
_user_specified_name
variable:D @

_output_shapes
:f
"
_user_specified_name
gradient
Ф
S
#__inference__update_step_xla_286004
gradient
variable:<|f*
_XlaMustCompile(*(
_construction_contextkEagerRuntime*#
_input_shapes
:<|f: *
	_noinline(:($
"
_user_specified_name
variable:L H
"
_output_shapes
:<|f
"
_user_specified_name
gradient
­
E
)__inference_restored_function_body_285129

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
N__inference_whiten_passthrough_50_layer_call_and_return_conditional_losses_833e
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
а

E__inference_conv1d_95_layer_call_and_return_conditional_losses_542198

inputsA
+conv1d_expanddims_1_readvariableop_resource:[t-
biasadd_readvariableop_resource:t
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
:[t*
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
:[tЌ
Conv1DConv2DConv1D/ExpandDims:output:0Conv1D/ExpandDims_1:output:0*
T0*/
_output_shapes
:џџџџџџџџџt*
paddingSAME*
strides
m
Conv1D/SqueezeSqueezeConv1D:output:0*
T0*+
_output_shapes
:џџџџџџџџџt*
squeeze_dims

§џџџџџџџџr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:t*
dtype0
BiasAddBiasAddConv1D/Squeeze:output:0BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:џџџџџџџџџtZ
SoftmaxSoftmaxBiasAdd:output:0*
T0*+
_output_shapes
:џџџџџџџџџtd
IdentityIdentitySoftmax:softmax:0^NoOp*
T0*+
_output_shapes
:џџџџџџџџџt
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
у

*__inference_dense_117_layer_call_fn_542987

inputs
unknown:X|
	unknown_0:|
identityЂStatefulPartitionedCallэ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:џџџџџџџџџ|*$
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
E__inference_dense_117_layer_call_and_return_conditional_losses_542281s
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:џџџџџџџџџ|`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:џџџџџџџџџX: : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:џџџџџџџџџX
 
_user_specified_nameinputs
р
В
#__inference_internal_grad_fn_543262
result_grads_0
result_grads_1
result_grads_2
mul_dense_116_beta
mul_dense_116_biasadd
identity

identity_1|
mulMulmul_dense_116_betamul_dense_116_biasadd^result_grads_0*
T0*+
_output_shapes
:џџџџџџџџџXQ
SigmoidSigmoidmul:z:0*
T0*+
_output_shapes
:џџџџџџџџџXm
mul_1Mulmul_dense_116_betamul_dense_116_biasadd*
T0*+
_output_shapes
:џџџџџџџџџXJ
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?]
subSubsub/x:output:0Sigmoid:y:0*
T0*+
_output_shapes
:џџџџџџџџџXV
mul_2Mul	mul_1:z:0sub:z:0*
T0*+
_output_shapes
:џџџџџџџџџXJ
add/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?]
addAddV2add/x:output:0	mul_2:z:0*
T0*+
_output_shapes
:џџџџџџџџџXX
mul_3MulSigmoid:y:0add:z:0*
T0*+
_output_shapes
:џџџџџџџџџX]
SquareSquaremul_dense_116_biasadd*
T0*+
_output_shapes
:џџџџџџџџџX^
mul_4Mulresult_grads_0
Square:y:0*
T0*+
_output_shapes
:џџџџџџџџџXZ
mul_5Mul	mul_4:z:0Sigmoid:y:0*
T0*+
_output_shapes
:џџџџџџџџџXL
sub_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?a
sub_1Subsub_1/x:output:0Sigmoid:y:0*
T0*+
_output_shapes
:џџџџџџџџџXX
mul_6Mul	mul_5:z:0	sub_1:z:0*
T0*+
_output_shapes
:џџџџџџџџџXZ
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
:џџџџџџџџџXU
IdentityIdentity	mul_7:z:0*
T0*+
_output_shapes
:џџџџџџџџџXE

Identity_1IdentitySum:output:0*
T0*
_output_shapes
: "!

identity_1Identity_1:output:0"
identityIdentity:output:0*\
_input_shapesK
I:џџџџџџџџџX:џџџџџџџџџX: : :џџџџџџџџџX:1-
+
_output_shapes
:џџџџџџџџџX:
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
:џџџџџџџџџX
(
_user_specified_nameresult_grads_1: 
&
 _has_manual_control_dependencies(
+
_output_shapes
:џџџџџџџџџX
(
_user_specified_nameresult_grads_0
И
O
#__inference__update_step_xla_285984
gradient
variable:tX*
_XlaMustCompile(*(
_construction_contextkEagerRuntime*
_input_shapes
:tX: *
	_noinline(:($
"
_user_specified_name
variable:H D

_output_shapes

:tX
"
_user_specified_name
gradient


$__inference_signature_wrapper_542628
	offsource
onsource
unknown:[t
	unknown_0:t
	unknown_1:tX
	unknown_2:X
	unknown_3:X|
	unknown_4:|
	unknown_5:<|f
	unknown_6:f
	unknown_7:f
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
!__inference__wrapped_model_542160o
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
о
_
C__inference_reshape_96_layer_call_and_return_conditional_losses_309

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
}
	
D__inference_model_96_layer_call_and_return_conditional_losses_542786
inputs_offsource
inputs_onsourceK
5conv1d_95_conv1d_expanddims_1_readvariableop_resource:[t7
)conv1d_95_biasadd_readvariableop_resource:t=
+dense_116_tensordot_readvariableop_resource:tX7
)dense_116_biasadd_readvariableop_resource:X=
+dense_117_tensordot_readvariableop_resource:X|7
)dense_117_biasadd_readvariableop_resource:|K
5conv1d_96_conv1d_expanddims_1_readvariableop_resource:<|f7
)conv1d_96_biasadd_readvariableop_resource:f@
.injection_masks_matmul_readvariableop_resource:f=
/injection_masks_biasadd_readvariableop_resource:
identityЂ&INJECTION_MASKS/BiasAdd/ReadVariableOpЂ%INJECTION_MASKS/MatMul/ReadVariableOpЂ conv1d_95/BiasAdd/ReadVariableOpЂ,conv1d_95/Conv1D/ExpandDims_1/ReadVariableOpЂ conv1d_96/BiasAdd/ReadVariableOpЂ,conv1d_96/Conv1D/ExpandDims_1/ReadVariableOpЂ dense_116/BiasAdd/ReadVariableOpЂ"dense_116/Tensordot/ReadVariableOpЂ dense_117/BiasAdd/ReadVariableOpЂ"dense_117/Tensordot/ReadVariableOpШ
%whiten_passthrough_50/PartitionedCallPartitionedCallinputs_offsource*
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
)__inference_restored_function_body_285129л
reshape_96/PartitionedCallPartitionedCall.whiten_passthrough_50/PartitionedCall:output:0*
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
)__inference_restored_function_body_285135j
conv1d_95/Conv1D/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
§џџџџџџџџГ
conv1d_95/Conv1D/ExpandDims
ExpandDims#reshape_96/PartitionedCall:output:0(conv1d_95/Conv1D/ExpandDims/dim:output:0*
T0*0
_output_shapes
:џџџџџџџџџІ
,conv1d_95/Conv1D/ExpandDims_1/ReadVariableOpReadVariableOp5conv1d_95_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:[t*
dtype0c
!conv1d_95/Conv1D/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : О
conv1d_95/Conv1D/ExpandDims_1
ExpandDims4conv1d_95/Conv1D/ExpandDims_1/ReadVariableOp:value:0*conv1d_95/Conv1D/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:[tЪ
conv1d_95/Conv1DConv2D$conv1d_95/Conv1D/ExpandDims:output:0&conv1d_95/Conv1D/ExpandDims_1:output:0*
T0*/
_output_shapes
:џџџџџџџџџt*
paddingSAME*
strides
m
conv1d_95/Conv1D/SqueezeSqueezeconv1d_95/Conv1D:output:0*
T0*+
_output_shapes
:џџџџџџџџџt*
squeeze_dims

§џџџџџџџџ
 conv1d_95/BiasAdd/ReadVariableOpReadVariableOp)conv1d_95_biasadd_readvariableop_resource*
_output_shapes
:t*
dtype0
conv1d_95/BiasAddBiasAdd!conv1d_95/Conv1D/Squeeze:output:0(conv1d_95/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:џџџџџџџџџtn
conv1d_95/SoftmaxSoftmaxconv1d_95/BiasAdd:output:0*
T0*+
_output_shapes
:џџџџџџџџџt
"dense_116/Tensordot/ReadVariableOpReadVariableOp+dense_116_tensordot_readvariableop_resource*
_output_shapes

:tX*
dtype0b
dense_116/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:i
dense_116/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       r
dense_116/Tensordot/ShapeShapeconv1d_95/Softmax:softmax:0*
T0*
_output_shapes
::эЯc
!dense_116/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : у
dense_116/Tensordot/GatherV2GatherV2"dense_116/Tensordot/Shape:output:0!dense_116/Tensordot/free:output:0*dense_116/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:e
#dense_116/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ч
dense_116/Tensordot/GatherV2_1GatherV2"dense_116/Tensordot/Shape:output:0!dense_116/Tensordot/axes:output:0,dense_116/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:c
dense_116/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 
dense_116/Tensordot/ProdProd%dense_116/Tensordot/GatherV2:output:0"dense_116/Tensordot/Const:output:0*
T0*
_output_shapes
: e
dense_116/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 
dense_116/Tensordot/Prod_1Prod'dense_116/Tensordot/GatherV2_1:output:0$dense_116/Tensordot/Const_1:output:0*
T0*
_output_shapes
: a
dense_116/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : Ф
dense_116/Tensordot/concatConcatV2!dense_116/Tensordot/free:output:0!dense_116/Tensordot/axes:output:0(dense_116/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:
dense_116/Tensordot/stackPack!dense_116/Tensordot/Prod:output:0#dense_116/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:Ђ
dense_116/Tensordot/transpose	Transposeconv1d_95/Softmax:softmax:0#dense_116/Tensordot/concat:output:0*
T0*+
_output_shapes
:џџџџџџџџџtЈ
dense_116/Tensordot/ReshapeReshape!dense_116/Tensordot/transpose:y:0"dense_116/Tensordot/stack:output:0*
T0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџЈ
dense_116/Tensordot/MatMulMatMul$dense_116/Tensordot/Reshape:output:0*dense_116/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџXe
dense_116/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:Xc
!dense_116/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : Я
dense_116/Tensordot/concat_1ConcatV2%dense_116/Tensordot/GatherV2:output:0$dense_116/Tensordot/Const_2:output:0*dense_116/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:Ё
dense_116/TensordotReshape$dense_116/Tensordot/MatMul:product:0%dense_116/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:џџџџџџџџџX
 dense_116/BiasAdd/ReadVariableOpReadVariableOp)dense_116_biasadd_readvariableop_resource*
_output_shapes
:X*
dtype0
dense_116/BiasAddBiasAdddense_116/Tensordot:output:0(dense_116/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:џџџџџџџџџXS
dense_116/betaConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
dense_116/mulMuldense_116/beta:output:0dense_116/BiasAdd:output:0*
T0*+
_output_shapes
:џџџџџџџџџXe
dense_116/SigmoidSigmoiddense_116/mul:z:0*
T0*+
_output_shapes
:џџџџџџџџџX
dense_116/mul_1Muldense_116/BiasAdd:output:0dense_116/Sigmoid:y:0*
T0*+
_output_shapes
:џџџџџџџџџXi
dense_116/IdentityIdentitydense_116/mul_1:z:0*
T0*+
_output_shapes
:џџџџџџџџџXь
dense_116/IdentityN	IdentityNdense_116/mul_1:z:0dense_116/BiasAdd:output:0dense_116/beta:output:0*
T
2*,
_gradient_op_typeCustomGradient-542725*D
_output_shapes2
0:џџџџџџџџџX:џџџџџџџџџX: b
 max_pooling1d_115/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :­
max_pooling1d_115/ExpandDims
ExpandDimsdense_116/IdentityN:output:0)max_pooling1d_115/ExpandDims/dim:output:0*
T0*/
_output_shapes
:џџџџџџџџџXЗ
max_pooling1d_115/MaxPoolMaxPool%max_pooling1d_115/ExpandDims:output:0*/
_output_shapes
:џџџџџџџџџX*
ksize
*
paddingSAME*
strides
	
max_pooling1d_115/SqueezeSqueeze"max_pooling1d_115/MaxPool:output:0*
T0*+
_output_shapes
:џџџџџџџџџX*
squeeze_dims

"dense_117/Tensordot/ReadVariableOpReadVariableOp+dense_117_tensordot_readvariableop_resource*
_output_shapes

:X|*
dtype0b
dense_117/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:i
dense_117/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       y
dense_117/Tensordot/ShapeShape"max_pooling1d_115/Squeeze:output:0*
T0*
_output_shapes
::эЯc
!dense_117/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : у
dense_117/Tensordot/GatherV2GatherV2"dense_117/Tensordot/Shape:output:0!dense_117/Tensordot/free:output:0*dense_117/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:e
#dense_117/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ч
dense_117/Tensordot/GatherV2_1GatherV2"dense_117/Tensordot/Shape:output:0!dense_117/Tensordot/axes:output:0,dense_117/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:c
dense_117/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 
dense_117/Tensordot/ProdProd%dense_117/Tensordot/GatherV2:output:0"dense_117/Tensordot/Const:output:0*
T0*
_output_shapes
: e
dense_117/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 
dense_117/Tensordot/Prod_1Prod'dense_117/Tensordot/GatherV2_1:output:0$dense_117/Tensordot/Const_1:output:0*
T0*
_output_shapes
: a
dense_117/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : Ф
dense_117/Tensordot/concatConcatV2!dense_117/Tensordot/free:output:0!dense_117/Tensordot/axes:output:0(dense_117/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:
dense_117/Tensordot/stackPack!dense_117/Tensordot/Prod:output:0#dense_117/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:Љ
dense_117/Tensordot/transpose	Transpose"max_pooling1d_115/Squeeze:output:0#dense_117/Tensordot/concat:output:0*
T0*+
_output_shapes
:џџџџџџџџџXЈ
dense_117/Tensordot/ReshapeReshape!dense_117/Tensordot/transpose:y:0"dense_117/Tensordot/stack:output:0*
T0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџЈ
dense_117/Tensordot/MatMulMatMul$dense_117/Tensordot/Reshape:output:0*dense_117/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ|e
dense_117/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:|c
!dense_117/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : Я
dense_117/Tensordot/concat_1ConcatV2%dense_117/Tensordot/GatherV2:output:0$dense_117/Tensordot/Const_2:output:0*dense_117/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:Ё
dense_117/TensordotReshape$dense_117/Tensordot/MatMul:product:0%dense_117/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:џџџџџџџџџ|
 dense_117/BiasAdd/ReadVariableOpReadVariableOp)dense_117_biasadd_readvariableop_resource*
_output_shapes
:|*
dtype0
dense_117/BiasAddBiasAdddense_117/Tensordot:output:0(dense_117/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:џџџџџџџџџ|h
dense_117/ReluReludense_117/BiasAdd:output:0*
T0*+
_output_shapes
:џџџџџџџџџ|j
conv1d_96/Conv1D/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
§џџџџџџџџЋ
conv1d_96/Conv1D/ExpandDims
ExpandDimsdense_117/Relu:activations:0(conv1d_96/Conv1D/ExpandDims/dim:output:0*
T0*/
_output_shapes
:џџџџџџџџџ|І
,conv1d_96/Conv1D/ExpandDims_1/ReadVariableOpReadVariableOp5conv1d_96_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:<|f*
dtype0c
!conv1d_96/Conv1D/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : О
conv1d_96/Conv1D/ExpandDims_1
ExpandDims4conv1d_96/Conv1D/ExpandDims_1/ReadVariableOp:value:0*conv1d_96/Conv1D/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:<|fЪ
conv1d_96/Conv1DConv2D$conv1d_96/Conv1D/ExpandDims:output:0&conv1d_96/Conv1D/ExpandDims_1:output:0*
T0*/
_output_shapes
:џџџџџџџџџf*
paddingSAME*
strides
B
conv1d_96/Conv1D/SqueezeSqueezeconv1d_96/Conv1D:output:0*
T0*+
_output_shapes
:џџџџџџџџџf*
squeeze_dims

§џџџџџџџџ
 conv1d_96/BiasAdd/ReadVariableOpReadVariableOp)conv1d_96_biasadd_readvariableop_resource*
_output_shapes
:f*
dtype0
conv1d_96/BiasAddBiasAdd!conv1d_96/Conv1D/Squeeze:output:0(conv1d_96/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:џџџџџџџџџff
conv1d_96/EluEluconv1d_96/BiasAdd:output:0*
T0*+
_output_shapes
:џџџџџџџџџfa
flatten_96/ConstConst*
_output_shapes
:*
dtype0*
valueB"џџџџf   
flatten_96/ReshapeReshapeconv1d_96/Elu:activations:0flatten_96/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџf
%INJECTION_MASKS/MatMul/ReadVariableOpReadVariableOp.injection_masks_matmul_readvariableop_resource*
_output_shapes

:f*
dtype0
INJECTION_MASKS/MatMulMatMulflatten_96/Reshape:output:0-INJECTION_MASKS/MatMul/ReadVariableOp:value:0*
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
:џџџџџџџџџЫ
NoOpNoOp'^INJECTION_MASKS/BiasAdd/ReadVariableOp&^INJECTION_MASKS/MatMul/ReadVariableOp!^conv1d_95/BiasAdd/ReadVariableOp-^conv1d_95/Conv1D/ExpandDims_1/ReadVariableOp!^conv1d_96/BiasAdd/ReadVariableOp-^conv1d_96/Conv1D/ExpandDims_1/ReadVariableOp!^dense_116/BiasAdd/ReadVariableOp#^dense_116/Tensordot/ReadVariableOp!^dense_117/BiasAdd/ReadVariableOp#^dense_117/Tensordot/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*X
_input_shapesG
E:џџџџџџџџџ:џџџџџџџџџ : : : : : : : : : : 2P
&INJECTION_MASKS/BiasAdd/ReadVariableOp&INJECTION_MASKS/BiasAdd/ReadVariableOp2N
%INJECTION_MASKS/MatMul/ReadVariableOp%INJECTION_MASKS/MatMul/ReadVariableOp2D
 conv1d_95/BiasAdd/ReadVariableOp conv1d_95/BiasAdd/ReadVariableOp2\
,conv1d_95/Conv1D/ExpandDims_1/ReadVariableOp,conv1d_95/Conv1D/ExpandDims_1/ReadVariableOp2D
 conv1d_96/BiasAdd/ReadVariableOp conv1d_96/BiasAdd/ReadVariableOp2\
,conv1d_96/Conv1D/ExpandDims_1/ReadVariableOp,conv1d_96/Conv1D/ExpandDims_1/ReadVariableOp2D
 dense_116/BiasAdd/ReadVariableOp dense_116/BiasAdd/ReadVariableOp2H
"dense_116/Tensordot/ReadVariableOp"dense_116/Tensordot/ReadVariableOp2D
 dense_117/BiasAdd/ReadVariableOp dense_117/BiasAdd/ReadVariableOp2H
"dense_117/Tensordot/ReadVariableOp"dense_117/Tensordot/ReadVariableOp:]Y
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
ђў
с
__inference__traced_save_543454
file_prefix5
read_disablecopyonread_kernel_4:[t-
read_1_disablecopyonread_bias_4:t3
!read_2_disablecopyonread_kernel_3:tX-
read_3_disablecopyonread_bias_3:X3
!read_4_disablecopyonread_kernel_2:X|-
read_5_disablecopyonread_bias_2:|7
!read_6_disablecopyonread_kernel_1:<|f-
read_7_disablecopyonread_bias_1:f1
read_8_disablecopyonread_kernel:f+
read_9_disablecopyonread_bias:-
#read_10_disablecopyonread_iteration:	 1
'read_11_disablecopyonread_learning_rate: ?
)read_12_disablecopyonread_adam_m_kernel_4:[t?
)read_13_disablecopyonread_adam_v_kernel_4:[t5
'read_14_disablecopyonread_adam_m_bias_4:t5
'read_15_disablecopyonread_adam_v_bias_4:t;
)read_16_disablecopyonread_adam_m_kernel_3:tX;
)read_17_disablecopyonread_adam_v_kernel_3:tX5
'read_18_disablecopyonread_adam_m_bias_3:X5
'read_19_disablecopyonread_adam_v_bias_3:X;
)read_20_disablecopyonread_adam_m_kernel_2:X|;
)read_21_disablecopyonread_adam_v_kernel_2:X|5
'read_22_disablecopyonread_adam_m_bias_2:|5
'read_23_disablecopyonread_adam_v_bias_2:|?
)read_24_disablecopyonread_adam_m_kernel_1:<|f?
)read_25_disablecopyonread_adam_v_kernel_1:<|f5
'read_26_disablecopyonread_adam_m_bias_1:f5
'read_27_disablecopyonread_adam_v_bias_1:f9
'read_28_disablecopyonread_adam_m_kernel:f9
'read_29_disablecopyonread_adam_v_kernel:f3
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
:[t*
dtype0m
IdentityIdentityRead/ReadVariableOp:value:0"/device:CPU:0*
T0*"
_output_shapes
:[te

Identity_1IdentityIdentity:output:0"/device:CPU:0*
T0*"
_output_shapes
:[ts
Read_1/DisableCopyOnReadDisableCopyOnReadread_1_disablecopyonread_bias_4"/device:CPU:0*
_output_shapes
 
Read_1/ReadVariableOpReadVariableOpread_1_disablecopyonread_bias_4^Read_1/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:t*
dtype0i

Identity_2IdentityRead_1/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:t_

Identity_3IdentityIdentity_2:output:0"/device:CPU:0*
T0*
_output_shapes
:tu
Read_2/DisableCopyOnReadDisableCopyOnRead!read_2_disablecopyonread_kernel_3"/device:CPU:0*
_output_shapes
 Ё
Read_2/ReadVariableOpReadVariableOp!read_2_disablecopyonread_kernel_3^Read_2/DisableCopyOnRead"/device:CPU:0*
_output_shapes

:tX*
dtype0m

Identity_4IdentityRead_2/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

:tXc

Identity_5IdentityIdentity_4:output:0"/device:CPU:0*
T0*
_output_shapes

:tXs
Read_3/DisableCopyOnReadDisableCopyOnReadread_3_disablecopyonread_bias_3"/device:CPU:0*
_output_shapes
 
Read_3/ReadVariableOpReadVariableOpread_3_disablecopyonread_bias_3^Read_3/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:X*
dtype0i

Identity_6IdentityRead_3/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:X_

Identity_7IdentityIdentity_6:output:0"/device:CPU:0*
T0*
_output_shapes
:Xu
Read_4/DisableCopyOnReadDisableCopyOnRead!read_4_disablecopyonread_kernel_2"/device:CPU:0*
_output_shapes
 Ё
Read_4/ReadVariableOpReadVariableOp!read_4_disablecopyonread_kernel_2^Read_4/DisableCopyOnRead"/device:CPU:0*
_output_shapes

:X|*
dtype0m

Identity_8IdentityRead_4/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

:X|c

Identity_9IdentityIdentity_8:output:0"/device:CPU:0*
T0*
_output_shapes

:X|s
Read_5/DisableCopyOnReadDisableCopyOnReadread_5_disablecopyonread_bias_2"/device:CPU:0*
_output_shapes
 
Read_5/ReadVariableOpReadVariableOpread_5_disablecopyonread_bias_2^Read_5/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:|*
dtype0j
Identity_10IdentityRead_5/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:|a
Identity_11IdentityIdentity_10:output:0"/device:CPU:0*
T0*
_output_shapes
:|u
Read_6/DisableCopyOnReadDisableCopyOnRead!read_6_disablecopyonread_kernel_1"/device:CPU:0*
_output_shapes
 Ѕ
Read_6/ReadVariableOpReadVariableOp!read_6_disablecopyonread_kernel_1^Read_6/DisableCopyOnRead"/device:CPU:0*"
_output_shapes
:<|f*
dtype0r
Identity_12IdentityRead_6/ReadVariableOp:value:0"/device:CPU:0*
T0*"
_output_shapes
:<|fi
Identity_13IdentityIdentity_12:output:0"/device:CPU:0*
T0*"
_output_shapes
:<|fs
Read_7/DisableCopyOnReadDisableCopyOnReadread_7_disablecopyonread_bias_1"/device:CPU:0*
_output_shapes
 
Read_7/ReadVariableOpReadVariableOpread_7_disablecopyonread_bias_1^Read_7/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:f*
dtype0j
Identity_14IdentityRead_7/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:fa
Identity_15IdentityIdentity_14:output:0"/device:CPU:0*
T0*
_output_shapes
:fs
Read_8/DisableCopyOnReadDisableCopyOnReadread_8_disablecopyonread_kernel"/device:CPU:0*
_output_shapes
 
Read_8/ReadVariableOpReadVariableOpread_8_disablecopyonread_kernel^Read_8/DisableCopyOnRead"/device:CPU:0*
_output_shapes

:f*
dtype0n
Identity_16IdentityRead_8/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

:fe
Identity_17IdentityIdentity_16:output:0"/device:CPU:0*
T0*
_output_shapes

:fq
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
:[t*
dtype0s
Identity_24IdentityRead_12/ReadVariableOp:value:0"/device:CPU:0*
T0*"
_output_shapes
:[ti
Identity_25IdentityIdentity_24:output:0"/device:CPU:0*
T0*"
_output_shapes
:[t~
Read_13/DisableCopyOnReadDisableCopyOnRead)read_13_disablecopyonread_adam_v_kernel_4"/device:CPU:0*
_output_shapes
 Џ
Read_13/ReadVariableOpReadVariableOp)read_13_disablecopyonread_adam_v_kernel_4^Read_13/DisableCopyOnRead"/device:CPU:0*"
_output_shapes
:[t*
dtype0s
Identity_26IdentityRead_13/ReadVariableOp:value:0"/device:CPU:0*
T0*"
_output_shapes
:[ti
Identity_27IdentityIdentity_26:output:0"/device:CPU:0*
T0*"
_output_shapes
:[t|
Read_14/DisableCopyOnReadDisableCopyOnRead'read_14_disablecopyonread_adam_m_bias_4"/device:CPU:0*
_output_shapes
 Ѕ
Read_14/ReadVariableOpReadVariableOp'read_14_disablecopyonread_adam_m_bias_4^Read_14/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:t*
dtype0k
Identity_28IdentityRead_14/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:ta
Identity_29IdentityIdentity_28:output:0"/device:CPU:0*
T0*
_output_shapes
:t|
Read_15/DisableCopyOnReadDisableCopyOnRead'read_15_disablecopyonread_adam_v_bias_4"/device:CPU:0*
_output_shapes
 Ѕ
Read_15/ReadVariableOpReadVariableOp'read_15_disablecopyonread_adam_v_bias_4^Read_15/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:t*
dtype0k
Identity_30IdentityRead_15/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:ta
Identity_31IdentityIdentity_30:output:0"/device:CPU:0*
T0*
_output_shapes
:t~
Read_16/DisableCopyOnReadDisableCopyOnRead)read_16_disablecopyonread_adam_m_kernel_3"/device:CPU:0*
_output_shapes
 Ћ
Read_16/ReadVariableOpReadVariableOp)read_16_disablecopyonread_adam_m_kernel_3^Read_16/DisableCopyOnRead"/device:CPU:0*
_output_shapes

:tX*
dtype0o
Identity_32IdentityRead_16/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

:tXe
Identity_33IdentityIdentity_32:output:0"/device:CPU:0*
T0*
_output_shapes

:tX~
Read_17/DisableCopyOnReadDisableCopyOnRead)read_17_disablecopyonread_adam_v_kernel_3"/device:CPU:0*
_output_shapes
 Ћ
Read_17/ReadVariableOpReadVariableOp)read_17_disablecopyonread_adam_v_kernel_3^Read_17/DisableCopyOnRead"/device:CPU:0*
_output_shapes

:tX*
dtype0o
Identity_34IdentityRead_17/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

:tXe
Identity_35IdentityIdentity_34:output:0"/device:CPU:0*
T0*
_output_shapes

:tX|
Read_18/DisableCopyOnReadDisableCopyOnRead'read_18_disablecopyonread_adam_m_bias_3"/device:CPU:0*
_output_shapes
 Ѕ
Read_18/ReadVariableOpReadVariableOp'read_18_disablecopyonread_adam_m_bias_3^Read_18/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:X*
dtype0k
Identity_36IdentityRead_18/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:Xa
Identity_37IdentityIdentity_36:output:0"/device:CPU:0*
T0*
_output_shapes
:X|
Read_19/DisableCopyOnReadDisableCopyOnRead'read_19_disablecopyonread_adam_v_bias_3"/device:CPU:0*
_output_shapes
 Ѕ
Read_19/ReadVariableOpReadVariableOp'read_19_disablecopyonread_adam_v_bias_3^Read_19/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:X*
dtype0k
Identity_38IdentityRead_19/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:Xa
Identity_39IdentityIdentity_38:output:0"/device:CPU:0*
T0*
_output_shapes
:X~
Read_20/DisableCopyOnReadDisableCopyOnRead)read_20_disablecopyonread_adam_m_kernel_2"/device:CPU:0*
_output_shapes
 Ћ
Read_20/ReadVariableOpReadVariableOp)read_20_disablecopyonread_adam_m_kernel_2^Read_20/DisableCopyOnRead"/device:CPU:0*
_output_shapes

:X|*
dtype0o
Identity_40IdentityRead_20/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

:X|e
Identity_41IdentityIdentity_40:output:0"/device:CPU:0*
T0*
_output_shapes

:X|~
Read_21/DisableCopyOnReadDisableCopyOnRead)read_21_disablecopyonread_adam_v_kernel_2"/device:CPU:0*
_output_shapes
 Ћ
Read_21/ReadVariableOpReadVariableOp)read_21_disablecopyonread_adam_v_kernel_2^Read_21/DisableCopyOnRead"/device:CPU:0*
_output_shapes

:X|*
dtype0o
Identity_42IdentityRead_21/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

:X|e
Identity_43IdentityIdentity_42:output:0"/device:CPU:0*
T0*
_output_shapes

:X||
Read_22/DisableCopyOnReadDisableCopyOnRead'read_22_disablecopyonread_adam_m_bias_2"/device:CPU:0*
_output_shapes
 Ѕ
Read_22/ReadVariableOpReadVariableOp'read_22_disablecopyonread_adam_m_bias_2^Read_22/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:|*
dtype0k
Identity_44IdentityRead_22/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:|a
Identity_45IdentityIdentity_44:output:0"/device:CPU:0*
T0*
_output_shapes
:||
Read_23/DisableCopyOnReadDisableCopyOnRead'read_23_disablecopyonread_adam_v_bias_2"/device:CPU:0*
_output_shapes
 Ѕ
Read_23/ReadVariableOpReadVariableOp'read_23_disablecopyonread_adam_v_bias_2^Read_23/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:|*
dtype0k
Identity_46IdentityRead_23/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:|a
Identity_47IdentityIdentity_46:output:0"/device:CPU:0*
T0*
_output_shapes
:|~
Read_24/DisableCopyOnReadDisableCopyOnRead)read_24_disablecopyonread_adam_m_kernel_1"/device:CPU:0*
_output_shapes
 Џ
Read_24/ReadVariableOpReadVariableOp)read_24_disablecopyonread_adam_m_kernel_1^Read_24/DisableCopyOnRead"/device:CPU:0*"
_output_shapes
:<|f*
dtype0s
Identity_48IdentityRead_24/ReadVariableOp:value:0"/device:CPU:0*
T0*"
_output_shapes
:<|fi
Identity_49IdentityIdentity_48:output:0"/device:CPU:0*
T0*"
_output_shapes
:<|f~
Read_25/DisableCopyOnReadDisableCopyOnRead)read_25_disablecopyonread_adam_v_kernel_1"/device:CPU:0*
_output_shapes
 Џ
Read_25/ReadVariableOpReadVariableOp)read_25_disablecopyonread_adam_v_kernel_1^Read_25/DisableCopyOnRead"/device:CPU:0*"
_output_shapes
:<|f*
dtype0s
Identity_50IdentityRead_25/ReadVariableOp:value:0"/device:CPU:0*
T0*"
_output_shapes
:<|fi
Identity_51IdentityIdentity_50:output:0"/device:CPU:0*
T0*"
_output_shapes
:<|f|
Read_26/DisableCopyOnReadDisableCopyOnRead'read_26_disablecopyonread_adam_m_bias_1"/device:CPU:0*
_output_shapes
 Ѕ
Read_26/ReadVariableOpReadVariableOp'read_26_disablecopyonread_adam_m_bias_1^Read_26/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:f*
dtype0k
Identity_52IdentityRead_26/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:fa
Identity_53IdentityIdentity_52:output:0"/device:CPU:0*
T0*
_output_shapes
:f|
Read_27/DisableCopyOnReadDisableCopyOnRead'read_27_disablecopyonread_adam_v_bias_1"/device:CPU:0*
_output_shapes
 Ѕ
Read_27/ReadVariableOpReadVariableOp'read_27_disablecopyonread_adam_v_bias_1^Read_27/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:f*
dtype0k
Identity_54IdentityRead_27/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:fa
Identity_55IdentityIdentity_54:output:0"/device:CPU:0*
T0*
_output_shapes
:f|
Read_28/DisableCopyOnReadDisableCopyOnRead'read_28_disablecopyonread_adam_m_kernel"/device:CPU:0*
_output_shapes
 Љ
Read_28/ReadVariableOpReadVariableOp'read_28_disablecopyonread_adam_m_kernel^Read_28/DisableCopyOnRead"/device:CPU:0*
_output_shapes

:f*
dtype0o
Identity_56IdentityRead_28/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

:fe
Identity_57IdentityIdentity_56:output:0"/device:CPU:0*
T0*
_output_shapes

:f|
Read_29/DisableCopyOnReadDisableCopyOnRead'read_29_disablecopyonread_adam_v_kernel"/device:CPU:0*
_output_shapes
 Љ
Read_29/ReadVariableOpReadVariableOp'read_29_disablecopyonread_adam_v_kernel^Read_29/DisableCopyOnRead"/device:CPU:0*
_output_shapes

:f*
dtype0o
Identity_58IdentityRead_29/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

:fe
Identity_59IdentityIdentity_58:output:0"/device:CPU:0*
T0*
_output_shapes

:fz
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
В
ќ
E__inference_dense_117_layer_call_and_return_conditional_losses_542281

inputs3
!tensordot_readvariableop_resource:X|-
biasadd_readvariableop_resource:|
identityЂBiasAdd/ReadVariableOpЂTensordot/ReadVariableOpz
Tensordot/ReadVariableOpReadVariableOp!tensordot_readvariableop_resource*
_output_shapes

:X|*
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
:џџџџџџџџџX
Tensordot/ReshapeReshapeTensordot/transpose:y:0Tensordot/stack:output:0*
T0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ
Tensordot/MatMulMatMulTensordot/Reshape:output:0 Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ|[
Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:|Y
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
:џџџџџџџџџ|r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:|*
dtype0|
BiasAddBiasAddTensordot:output:0BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:џџџџџџџџџ|T
ReluReluBiasAdd:output:0*
T0*+
_output_shapes
:џџџџџџџџџ|e
IdentityIdentityRelu:activations:0^NoOp*
T0*+
_output_shapes
:џџџџџџџџџ|z
NoOpNoOp^BiasAdd/ReadVariableOp^Tensordot/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:џџџџџџџџџX: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp24
Tensordot/ReadVariableOpTensordot/ReadVariableOp:S O
+
_output_shapes
:џџџџџџџџџX
 
_user_specified_nameinputs
б
i
M__inference_max_pooling1d_115_layer_call_and_return_conditional_losses_542169

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
*
paddingSAME*
strides
	
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
и
P
4__inference_whiten_passthrough_50_layer_call_fn_1101

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
O__inference_whiten_passthrough_50_layer_call_and_return_conditional_losses_1096e
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
Ш
B
__inference_crop_samples_816
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
с)

D__inference_model_96_layer_call_and_return_conditional_losses_542467
inputs_1

inputs&
conv1d_95_542439:[t
conv1d_95_542441:t"
dense_116_542444:tX
dense_116_542446:X"
dense_117_542450:X|
dense_117_542452:|&
conv1d_96_542455:<|f
conv1d_96_542457:f(
injection_masks_542461:f$
injection_masks_542463:
identityЂ'INJECTION_MASKS/StatefulPartitionedCallЂ!conv1d_95/StatefulPartitionedCallЂ!conv1d_96/StatefulPartitionedCallЂ!dense_116/StatefulPartitionedCallЂ!dense_117/StatefulPartitionedCallР
%whiten_passthrough_50/PartitionedCallPartitionedCallinputs_1*
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
)__inference_restored_function_body_285129л
reshape_96/PartitionedCallPartitionedCall.whiten_passthrough_50/PartitionedCall:output:0*
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
)__inference_restored_function_body_285135Є
!conv1d_95/StatefulPartitionedCallStatefulPartitionedCall#reshape_96/PartitionedCall:output:0conv1d_95_542439conv1d_95_542441*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:џџџџџџџџџt*$
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
E__inference_conv1d_95_layer_call_and_return_conditional_losses_542198Ћ
!dense_116/StatefulPartitionedCallStatefulPartitionedCall*conv1d_95/StatefulPartitionedCall:output:0dense_116_542444dense_116_542446*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:џџџџџџџџџX*$
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
E__inference_dense_116_layer_call_and_return_conditional_losses_542243
!max_pooling1d_115/PartitionedCallPartitionedCall*dense_116/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:џџџџџџџџџX* 
_read_only_resource_inputs
 *<
config_proto,*

CPU

GPU(2*0J

  zE8 *V
fQRO
M__inference_max_pooling1d_115_layer_call_and_return_conditional_losses_542169Ћ
!dense_117/StatefulPartitionedCallStatefulPartitionedCall*max_pooling1d_115/PartitionedCall:output:0dense_117_542450dense_117_542452*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:џџџџџџџџџ|*$
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
E__inference_dense_117_layer_call_and_return_conditional_losses_542281Ћ
!conv1d_96/StatefulPartitionedCallStatefulPartitionedCall*dense_117/StatefulPartitionedCall:output:0conv1d_96_542455conv1d_96_542457*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:џџџџџџџџџf*$
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
E__inference_conv1d_96_layer_call_and_return_conditional_losses_542303я
flatten_96/PartitionedCallPartitionedCall*conv1d_96/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџf* 
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
F__inference_flatten_96_layer_call_and_return_conditional_losses_542315И
'INJECTION_MASKS/StatefulPartitionedCallStatefulPartitionedCall#flatten_96/PartitionedCall:output:0injection_masks_542461injection_masks_542463*
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
K__inference_INJECTION_MASKS_layer_call_and_return_conditional_losses_542328
IdentityIdentity0INJECTION_MASKS/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ
NoOpNoOp(^INJECTION_MASKS/StatefulPartitionedCall"^conv1d_95/StatefulPartitionedCall"^conv1d_96/StatefulPartitionedCall"^dense_116/StatefulPartitionedCall"^dense_117/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*X
_input_shapesG
E:џџџџџџџџџ:џџџџџџџџџ : : : : : : : : : : 2R
'INJECTION_MASKS/StatefulPartitionedCall'INJECTION_MASKS/StatefulPartitionedCall2F
!conv1d_95/StatefulPartitionedCall!conv1d_95/StatefulPartitionedCall2F
!conv1d_96/StatefulPartitionedCall!conv1d_96/StatefulPartitionedCall2F
!dense_116/StatefulPartitionedCall!dense_116/StatefulPartitionedCall2F
!dense_117/StatefulPartitionedCall!dense_117/StatefulPartitionedCall:TP
,
_output_shapes
:џџџџџџџџџ 
 
_user_specified_nameinputs:U Q
-
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
ѕ
k
O__inference_whiten_passthrough_50_layer_call_and_return_conditional_losses_1096

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
__inference_crop_samples_816I
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
 
E
)__inference_restored_function_body_285135

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
C__inference_reshape_96_layer_call_and_return_conditional_losses_309e
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
ъ)

D__inference_model_96_layer_call_and_return_conditional_losses_542335
	offsource
onsource&
conv1d_95_542199:[t
conv1d_95_542201:t"
dense_116_542244:tX
dense_116_542246:X"
dense_117_542282:X|
dense_117_542284:|&
conv1d_96_542304:<|f
conv1d_96_542306:f(
injection_masks_542329:f$
injection_masks_542331:
identityЂ'INJECTION_MASKS/StatefulPartitionedCallЂ!conv1d_95/StatefulPartitionedCallЂ!conv1d_96/StatefulPartitionedCallЂ!dense_116/StatefulPartitionedCallЂ!dense_117/StatefulPartitionedCallС
%whiten_passthrough_50/PartitionedCallPartitionedCall	offsource*
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
)__inference_restored_function_body_285129л
reshape_96/PartitionedCallPartitionedCall.whiten_passthrough_50/PartitionedCall:output:0*
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
)__inference_restored_function_body_285135Є
!conv1d_95/StatefulPartitionedCallStatefulPartitionedCall#reshape_96/PartitionedCall:output:0conv1d_95_542199conv1d_95_542201*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:џџџџџџџџџt*$
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
E__inference_conv1d_95_layer_call_and_return_conditional_losses_542198Ћ
!dense_116/StatefulPartitionedCallStatefulPartitionedCall*conv1d_95/StatefulPartitionedCall:output:0dense_116_542244dense_116_542246*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:џџџџџџџџџX*$
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
E__inference_dense_116_layer_call_and_return_conditional_losses_542243
!max_pooling1d_115/PartitionedCallPartitionedCall*dense_116/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:џџџџџџџџџX* 
_read_only_resource_inputs
 *<
config_proto,*

CPU

GPU(2*0J

  zE8 *V
fQRO
M__inference_max_pooling1d_115_layer_call_and_return_conditional_losses_542169Ћ
!dense_117/StatefulPartitionedCallStatefulPartitionedCall*max_pooling1d_115/PartitionedCall:output:0dense_117_542282dense_117_542284*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:џџџџџџџџџ|*$
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
E__inference_dense_117_layer_call_and_return_conditional_losses_542281Ћ
!conv1d_96/StatefulPartitionedCallStatefulPartitionedCall*dense_117/StatefulPartitionedCall:output:0conv1d_96_542304conv1d_96_542306*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:џџџџџџџџџf*$
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
E__inference_conv1d_96_layer_call_and_return_conditional_losses_542303я
flatten_96/PartitionedCallPartitionedCall*conv1d_96/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџf* 
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
F__inference_flatten_96_layer_call_and_return_conditional_losses_542315И
'INJECTION_MASKS/StatefulPartitionedCallStatefulPartitionedCall#flatten_96/PartitionedCall:output:0injection_masks_542329injection_masks_542331*
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
K__inference_INJECTION_MASKS_layer_call_and_return_conditional_losses_542328
IdentityIdentity0INJECTION_MASKS/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ
NoOpNoOp(^INJECTION_MASKS/StatefulPartitionedCall"^conv1d_95/StatefulPartitionedCall"^conv1d_96/StatefulPartitionedCall"^dense_116/StatefulPartitionedCall"^dense_117/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*X
_input_shapesG
E:џџџџџџџџџ:џџџџџџџџџ : : : : : : : : : : 2R
'INJECTION_MASKS/StatefulPartitionedCall'INJECTION_MASKS/StatefulPartitionedCall2F
!conv1d_95/StatefulPartitionedCall!conv1d_95/StatefulPartitionedCall2F
!conv1d_96/StatefulPartitionedCall!conv1d_96/StatefulPartitionedCall2F
!dense_116/StatefulPartitionedCall!dense_116/StatefulPartitionedCall2F
!dense_117/StatefulPartitionedCall!dense_117/StatefulPartitionedCall:VR
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
Х

E__inference_conv1d_96_layer_call_and_return_conditional_losses_543043

inputsA
+conv1d_expanddims_1_readvariableop_resource:<|f-
biasadd_readvariableop_resource:f
identityЂBiasAdd/ReadVariableOpЂ"Conv1D/ExpandDims_1/ReadVariableOp`
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
:џџџџџџџџџ|
"Conv1D/ExpandDims_1/ReadVariableOpReadVariableOp+conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:<|f*
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
:<|fЌ
Conv1DConv2DConv1D/ExpandDims:output:0Conv1D/ExpandDims_1:output:0*
T0*/
_output_shapes
:џџџџџџџџџf*
paddingSAME*
strides
B
Conv1D/SqueezeSqueezeConv1D:output:0*
T0*+
_output_shapes
:џџџџџџџџџf*
squeeze_dims

§џџџџџџџџr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:f*
dtype0
BiasAddBiasAddConv1D/Squeeze:output:0BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:џџџџџџџџџfR
EluEluBiasAdd:output:0*
T0*+
_output_shapes
:џџџџџџџџџfd
IdentityIdentityElu:activations:0^NoOp*
T0*+
_output_shapes
:џџџџџџџџџf
NoOpNoOp^BiasAdd/ReadVariableOp#^Conv1D/ExpandDims_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:џџџџџџџџџ|: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2H
"Conv1D/ExpandDims_1/ReadVariableOp"Conv1D/ExpandDims_1/ReadVariableOp:S O
+
_output_shapes
:џџџџџџџџџ|
 
_user_specified_nameinputs
ц!
ў
E__inference_dense_116_layer_call_and_return_conditional_losses_542965

inputs3
!tensordot_readvariableop_resource:tX-
biasadd_readvariableop_resource:X

identity_1ЂBiasAdd/ReadVariableOpЂTensordot/ReadVariableOpz
Tensordot/ReadVariableOpReadVariableOp!tensordot_readvariableop_resource*
_output_shapes

:tX*
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
:џџџџџџџџџt
Tensordot/ReshapeReshapeTensordot/transpose:y:0Tensordot/stack:output:0*
T0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ
Tensordot/MatMulMatMulTensordot/Reshape:output:0 Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџX[
Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:XY
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
:џџџџџџџџџXr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:X*
dtype0|
BiasAddBiasAddTensordot:output:0BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:џџџџџџџџџXI
betaConst*
_output_shapes
: *
dtype0*
valueB
 *  ?a
mulMulbeta:output:0BiasAdd:output:0*
T0*+
_output_shapes
:џџџџџџџџџXQ
SigmoidSigmoidmul:z:0*
T0*+
_output_shapes
:џџџџџџџџџXa
mul_1MulBiasAdd:output:0Sigmoid:y:0*
T0*+
_output_shapes
:џџџџџџџџџXU
IdentityIdentity	mul_1:z:0*
T0*+
_output_shapes
:џџџџџџџџџXФ
	IdentityN	IdentityN	mul_1:z:0BiasAdd:output:0beta:output:0*
T
2*,
_gradient_op_typeCustomGradient-542956*D
_output_shapes2
0:џџџџџџџџџX:џџџџџџџџџX: g

Identity_1IdentityIdentityN:output:0^NoOp*
T0*+
_output_shapes
:џџџџџџџџџXz
NoOpNoOp^BiasAdd/ReadVariableOp^Tensordot/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:џџџџџџџџџt: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp24
Tensordot/ReadVariableOpTensordot/ReadVariableOp:S O
+
_output_shapes
:џџџџџџџџџt
 
_user_specified_nameinputs
Ќ
K
#__inference__update_step_xla_285989
gradient
variable:X*
_XlaMustCompile(*(
_construction_contextkEagerRuntime*
_input_shapes

:X: *
	_noinline(:($
"
_user_specified_name
variable:D @

_output_shapes
:X
"
_user_specified_name
gradient
Р
E
)__inference_reshape_96_layer_call_fn_1449

inputs
identityУ
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
 @E8 *M
fHRF
D__inference_reshape_96_layer_call_and_return_conditional_losses_1444e
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
#__inference_internal_grad_fn_543234
result_grads_0
result_grads_1
result_grads_2
mul_dense_116_beta
mul_dense_116_biasadd
identity

identity_1|
mulMulmul_dense_116_betamul_dense_116_biasadd^result_grads_0*
T0*+
_output_shapes
:џџџџџџџџџXQ
SigmoidSigmoidmul:z:0*
T0*+
_output_shapes
:џџџџџџџџџXm
mul_1Mulmul_dense_116_betamul_dense_116_biasadd*
T0*+
_output_shapes
:џџџџџџџџџXJ
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?]
subSubsub/x:output:0Sigmoid:y:0*
T0*+
_output_shapes
:џџџџџџџџџXV
mul_2Mul	mul_1:z:0sub:z:0*
T0*+
_output_shapes
:џџџџџџџџџXJ
add/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?]
addAddV2add/x:output:0	mul_2:z:0*
T0*+
_output_shapes
:џџџџџџџџџXX
mul_3MulSigmoid:y:0add:z:0*
T0*+
_output_shapes
:џџџџџџџџџX]
SquareSquaremul_dense_116_biasadd*
T0*+
_output_shapes
:џџџџџџџџџX^
mul_4Mulresult_grads_0
Square:y:0*
T0*+
_output_shapes
:џџџџџџџџџXZ
mul_5Mul	mul_4:z:0Sigmoid:y:0*
T0*+
_output_shapes
:џџџџџџџџџXL
sub_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?a
sub_1Subsub_1/x:output:0Sigmoid:y:0*
T0*+
_output_shapes
:џџџџџџџџџXX
mul_6Mul	mul_5:z:0	sub_1:z:0*
T0*+
_output_shapes
:џџџџџџџџџXZ
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
:џџџџџџџџџXU
IdentityIdentity	mul_7:z:0*
T0*+
_output_shapes
:џџџџџџџџџXE

Identity_1IdentitySum:output:0*
T0*
_output_shapes
: "!

identity_1Identity_1:output:0"
identityIdentity:output:0*\
_input_shapesK
I:џџџџџџџџџX:џџџџџџџџџX: : :џџџџџџџџџX:1-
+
_output_shapes
:џџџџџџџџџX:
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
:џџџџџџџџџX
(
_user_specified_nameresult_grads_1: 
&
 _has_manual_control_dependencies(
+
_output_shapes
:џџџџџџџџџX
(
_user_specified_nameresult_grads_0
с)

D__inference_model_96_layer_call_and_return_conditional_losses_542407
inputs_1

inputs&
conv1d_95_542379:[t
conv1d_95_542381:t"
dense_116_542384:tX
dense_116_542386:X"
dense_117_542390:X|
dense_117_542392:|&
conv1d_96_542395:<|f
conv1d_96_542397:f(
injection_masks_542401:f$
injection_masks_542403:
identityЂ'INJECTION_MASKS/StatefulPartitionedCallЂ!conv1d_95/StatefulPartitionedCallЂ!conv1d_96/StatefulPartitionedCallЂ!dense_116/StatefulPartitionedCallЂ!dense_117/StatefulPartitionedCallР
%whiten_passthrough_50/PartitionedCallPartitionedCallinputs_1*
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
)__inference_restored_function_body_285129л
reshape_96/PartitionedCallPartitionedCall.whiten_passthrough_50/PartitionedCall:output:0*
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
)__inference_restored_function_body_285135Є
!conv1d_95/StatefulPartitionedCallStatefulPartitionedCall#reshape_96/PartitionedCall:output:0conv1d_95_542379conv1d_95_542381*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:џџџџџџџџџt*$
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
E__inference_conv1d_95_layer_call_and_return_conditional_losses_542198Ћ
!dense_116/StatefulPartitionedCallStatefulPartitionedCall*conv1d_95/StatefulPartitionedCall:output:0dense_116_542384dense_116_542386*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:џџџџџџџџџX*$
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
E__inference_dense_116_layer_call_and_return_conditional_losses_542243
!max_pooling1d_115/PartitionedCallPartitionedCall*dense_116/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:џџџџџџџџџX* 
_read_only_resource_inputs
 *<
config_proto,*

CPU

GPU(2*0J

  zE8 *V
fQRO
M__inference_max_pooling1d_115_layer_call_and_return_conditional_losses_542169Ћ
!dense_117/StatefulPartitionedCallStatefulPartitionedCall*max_pooling1d_115/PartitionedCall:output:0dense_117_542390dense_117_542392*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:џџџџџџџџџ|*$
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
E__inference_dense_117_layer_call_and_return_conditional_losses_542281Ћ
!conv1d_96/StatefulPartitionedCallStatefulPartitionedCall*dense_117/StatefulPartitionedCall:output:0conv1d_96_542395conv1d_96_542397*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:џџџџџџџџџf*$
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
E__inference_conv1d_96_layer_call_and_return_conditional_losses_542303я
flatten_96/PartitionedCallPartitionedCall*conv1d_96/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџf* 
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
F__inference_flatten_96_layer_call_and_return_conditional_losses_542315И
'INJECTION_MASKS/StatefulPartitionedCallStatefulPartitionedCall#flatten_96/PartitionedCall:output:0injection_masks_542401injection_masks_542403*
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
K__inference_INJECTION_MASKS_layer_call_and_return_conditional_losses_542328
IdentityIdentity0INJECTION_MASKS/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ
NoOpNoOp(^INJECTION_MASKS/StatefulPartitionedCall"^conv1d_95/StatefulPartitionedCall"^conv1d_96/StatefulPartitionedCall"^dense_116/StatefulPartitionedCall"^dense_117/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*X
_input_shapesG
E:џџџџџџџџџ:џџџџџџџџџ : : : : : : : : : : 2R
'INJECTION_MASKS/StatefulPartitionedCall'INJECTION_MASKS/StatefulPartitionedCall2F
!conv1d_95/StatefulPartitionedCall!conv1d_95/StatefulPartitionedCall2F
!conv1d_96/StatefulPartitionedCall!conv1d_96/StatefulPartitionedCall2F
!dense_116/StatefulPartitionedCall!dense_116/StatefulPartitionedCall2F
!dense_117/StatefulPartitionedCall!dense_117/StatefulPartitionedCall:TP
,
_output_shapes
:џџџџџџџџџ 
 
_user_specified_nameinputs:U Q
-
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
а

E__inference_conv1d_95_layer_call_and_return_conditional_losses_542917

inputsA
+conv1d_expanddims_1_readvariableop_resource:[t-
biasadd_readvariableop_resource:t
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
:[t*
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
:[tЌ
Conv1DConv2DConv1D/ExpandDims:output:0Conv1D/ExpandDims_1:output:0*
T0*/
_output_shapes
:џџџџџџџџџt*
paddingSAME*
strides
m
Conv1D/SqueezeSqueezeConv1D:output:0*
T0*+
_output_shapes
:џџџџџџџџџt*
squeeze_dims

§џџџџџџџџr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:t*
dtype0
BiasAddBiasAddConv1D/Squeeze:output:0BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:џџџџџџџџџtZ
SoftmaxSoftmaxBiasAdd:output:0*
T0*+
_output_shapes
:џџџџџџџџџtd
IdentityIdentitySoftmax:softmax:0^NoOp*
T0*+
_output_shapes
:џџџџџџџџџt
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
 
Ф
#__inference_internal_grad_fn_543290
result_grads_0
result_grads_1
result_grads_2
mul_model_96_dense_116_beta"
mul_model_96_dense_116_biasadd
identity

identity_1
mulMulmul_model_96_dense_116_betamul_model_96_dense_116_biasadd^result_grads_0*
T0*+
_output_shapes
:џџџџџџџџџXQ
SigmoidSigmoidmul:z:0*
T0*+
_output_shapes
:џџџџџџџџџX
mul_1Mulmul_model_96_dense_116_betamul_model_96_dense_116_biasadd*
T0*+
_output_shapes
:џџџџџџџџџXJ
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?]
subSubsub/x:output:0Sigmoid:y:0*
T0*+
_output_shapes
:џџџџџџџџџXV
mul_2Mul	mul_1:z:0sub:z:0*
T0*+
_output_shapes
:џџџџџџџџџXJ
add/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?]
addAddV2add/x:output:0	mul_2:z:0*
T0*+
_output_shapes
:џџџџџџџџџXX
mul_3MulSigmoid:y:0add:z:0*
T0*+
_output_shapes
:џџџџџџџџџXf
SquareSquaremul_model_96_dense_116_biasadd*
T0*+
_output_shapes
:џџџџџџџџџX^
mul_4Mulresult_grads_0
Square:y:0*
T0*+
_output_shapes
:џџџџџџџџџXZ
mul_5Mul	mul_4:z:0Sigmoid:y:0*
T0*+
_output_shapes
:џџџџџџџџџXL
sub_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?a
sub_1Subsub_1/x:output:0Sigmoid:y:0*
T0*+
_output_shapes
:џџџџџџџџџXX
mul_6Mul	mul_5:z:0	sub_1:z:0*
T0*+
_output_shapes
:џџџџџџџџџXZ
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
:џџџџџџџџџXU
IdentityIdentity	mul_7:z:0*
T0*+
_output_shapes
:џџџџџџџџџXE

Identity_1IdentitySum:output:0*
T0*
_output_shapes
: "!

identity_1Identity_1:output:0"
identityIdentity:output:0*\
_input_shapesK
I:џџџџџџџџџX:џџџџџџџџџX: : :џџџџџџџџџX:1-
+
_output_shapes
:џџџџџџџџџX:
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
:џџџџџџџџџX
(
_user_specified_nameresult_grads_1: 
&
 _has_manual_control_dependencies(
+
_output_shapes
:џџџџџџџџџX
(
_user_specified_nameresult_grads_0
ц!
ў
E__inference_dense_116_layer_call_and_return_conditional_losses_542243

inputs3
!tensordot_readvariableop_resource:tX-
biasadd_readvariableop_resource:X

identity_1ЂBiasAdd/ReadVariableOpЂTensordot/ReadVariableOpz
Tensordot/ReadVariableOpReadVariableOp!tensordot_readvariableop_resource*
_output_shapes

:tX*
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
:џџџџџџџџџt
Tensordot/ReshapeReshapeTensordot/transpose:y:0Tensordot/stack:output:0*
T0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ
Tensordot/MatMulMatMulTensordot/Reshape:output:0 Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџX[
Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:XY
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
:џџџџџџџџџXr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:X*
dtype0|
BiasAddBiasAddTensordot:output:0BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:џџџџџџџџџXI
betaConst*
_output_shapes
: *
dtype0*
valueB
 *  ?a
mulMulbeta:output:0BiasAdd:output:0*
T0*+
_output_shapes
:џџџџџџџџџXQ
SigmoidSigmoidmul:z:0*
T0*+
_output_shapes
:џџџџџџџџџXa
mul_1MulBiasAdd:output:0Sigmoid:y:0*
T0*+
_output_shapes
:џџџџџџџџџXU
IdentityIdentity	mul_1:z:0*
T0*+
_output_shapes
:џџџџџџџџџXФ
	IdentityN	IdentityN	mul_1:z:0BiasAdd:output:0beta:output:0*
T
2*,
_gradient_op_typeCustomGradient-542234*D
_output_shapes2
0:џџџџџџџџџX:џџџџџџџџџX: g

Identity_1IdentityIdentityN:output:0^NoOp*
T0*+
_output_shapes
:џџџџџџџџџXz
NoOpNoOp^BiasAdd/ReadVariableOp^Tensordot/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:џџџџџџџџџt: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp24
Tensordot/ReadVariableOpTensordot/ReadVariableOp:S O
+
_output_shapes
:џџџџџџџџџt
 
_user_specified_nameinputs
Ч

)__inference_model_96_layer_call_fn_542430
	offsource
onsource
unknown:[t
	unknown_0:t
	unknown_1:tX
	unknown_2:X
	unknown_3:X|
	unknown_4:|
	unknown_5:<|f
	unknown_6:f
	unknown_7:f
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
D__inference_model_96_layer_call_and_return_conditional_losses_542407o
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
щ

*__inference_conv1d_95_layer_call_fn_542901

inputs
unknown:[t
	unknown_0:t
identityЂStatefulPartitionedCallэ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:џџџџџџџџџt*$
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
E__inference_conv1d_95_layer_call_and_return_conditional_losses_542198s
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:џџџџџџџџџt`
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
Ч

)__inference_model_96_layer_call_fn_542490
	offsource
onsource
unknown:[t
	unknown_0:t
	unknown_1:tX
	unknown_2:X
	unknown_3:X|
	unknown_4:|
	unknown_5:<|f
	unknown_6:f
	unknown_7:f
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
D__inference_model_96_layer_call_and_return_conditional_losses_542467o
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
ч

*__inference_conv1d_96_layer_call_fn_543027

inputs
unknown:<|f
	unknown_0:f
identityЂStatefulPartitionedCallэ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:џџџџџџџџџf*$
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
E__inference_conv1d_96_layer_call_and_return_conditional_losses_542303s
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:џџџџџџџџџf`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:џџџџџџџџџ|: : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:џџџџџџџџџ|
 
_user_specified_nameinputs

N
2__inference_max_pooling1d_115_layer_call_fn_542970

inputs
identityн
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
  zE8 *V
fQRO
M__inference_max_pooling1d_115_layer_call_and_return_conditional_losses_542169v
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
О
b
F__inference_flatten_96_layer_call_and_return_conditional_losses_542315

inputs
identityV
ConstConst*
_output_shapes
:*
dtype0*
valueB"џџџџf   \
ReshapeReshapeinputsConst:output:0*
T0*'
_output_shapes
:џџџџџџџџџfX
IdentityIdentityReshape:output:0*
T0*'
_output_shapes
:џџџџџџџџџf"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:џџџџџџџџџf:S O
+
_output_shapes
:џџџџџџџџџf
 
_user_specified_nameinputs
у

*__inference_dense_116_layer_call_fn_542926

inputs
unknown:tX
	unknown_0:X
identityЂStatefulPartitionedCallэ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:џџџџџџџџџX*$
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
E__inference_dense_116_layer_call_and_return_conditional_losses_542243s
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:џџџџџџџџџX`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:џџџџџџџџџt: : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:џџџџџџџџџt
 
_user_specified_nameinputs
ё

)__inference_model_96_layer_call_fn_542680
inputs_offsource
inputs_onsource
unknown:[t
	unknown_0:t
	unknown_1:tX
	unknown_2:X
	unknown_3:X|
	unknown_4:|
	unknown_5:<|f
	unknown_6:f
	unknown_7:f
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
D__inference_model_96_layer_call_and_return_conditional_losses_542467o
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
Ќ
K
#__inference__update_step_xla_285999
gradient
variable:|*
_XlaMustCompile(*(
_construction_contextkEagerRuntime*
_input_shapes

:|: *
	_noinline(:($
"
_user_specified_name
variable:D @

_output_shapes
:|
"
_user_specified_name
gradient
В
ќ
E__inference_dense_117_layer_call_and_return_conditional_losses_543018

inputs3
!tensordot_readvariableop_resource:X|-
biasadd_readvariableop_resource:|
identityЂBiasAdd/ReadVariableOpЂTensordot/ReadVariableOpz
Tensordot/ReadVariableOpReadVariableOp!tensordot_readvariableop_resource*
_output_shapes

:X|*
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
:џџџџџџџџџX
Tensordot/ReshapeReshapeTensordot/transpose:y:0Tensordot/stack:output:0*
T0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ
Tensordot/MatMulMatMulTensordot/Reshape:output:0 Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ|[
Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:|Y
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
:џџџџџџџџџ|r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:|*
dtype0|
BiasAddBiasAddTensordot:output:0BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:џџџџџџџџџ|T
ReluReluBiasAdd:output:0*
T0*+
_output_shapes
:џџџџџџџџџ|e
IdentityIdentityRelu:activations:0^NoOp*
T0*+
_output_shapes
:џџџџџџџџџ|z
NoOpNoOp^BiasAdd/ReadVariableOp^Tensordot/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:џџџџџџџџџX: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp24
Tensordot/ReadVariableOpTensordot/ReadVariableOp:S O
+
_output_shapes
:џџџџџџџџџX
 
_user_specified_nameinputs
Ќ
K
#__inference__update_step_xla_285979
gradient
variable:t*
_XlaMustCompile(*(
_construction_contextkEagerRuntime*
_input_shapes

:t: *
	_noinline(:($
"
_user_specified_name
variable:D @

_output_shapes
:t
"
_user_specified_name
gradient
О
b
F__inference_flatten_96_layer_call_and_return_conditional_losses_543054

inputs
identityV
ConstConst*
_output_shapes
:*
dtype0*
valueB"џџџџf   \
ReshapeReshapeinputsConst:output:0*
T0*'
_output_shapes
:џџџџџџџџџfX
IdentityIdentityReshape:output:0*
T0*'
_output_shapes
:џџџџџџџџџf"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:џџџџџџџџџf:S O
+
_output_shapes
:џџџџџџџџџf
 
_user_specified_nameinputs
Ё

ќ
K__inference_INJECTION_MASKS_layer_call_and_return_conditional_losses_543074

inputs0
matmul_readvariableop_resource:f-
biasadd_readvariableop_resource:
identityЂBiasAdd/ReadVariableOpЂMatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:f*
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
:џџџџџџџџџf: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:џџџџџџџџџf
 
_user_specified_nameinputs
ъ)

D__inference_model_96_layer_call_and_return_conditional_losses_542369
	offsource
onsource&
conv1d_95_542341:[t
conv1d_95_542343:t"
dense_116_542346:tX
dense_116_542348:X"
dense_117_542352:X|
dense_117_542354:|&
conv1d_96_542357:<|f
conv1d_96_542359:f(
injection_masks_542363:f$
injection_masks_542365:
identityЂ'INJECTION_MASKS/StatefulPartitionedCallЂ!conv1d_95/StatefulPartitionedCallЂ!conv1d_96/StatefulPartitionedCallЂ!dense_116/StatefulPartitionedCallЂ!dense_117/StatefulPartitionedCallС
%whiten_passthrough_50/PartitionedCallPartitionedCall	offsource*
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
)__inference_restored_function_body_285129л
reshape_96/PartitionedCallPartitionedCall.whiten_passthrough_50/PartitionedCall:output:0*
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
)__inference_restored_function_body_285135Є
!conv1d_95/StatefulPartitionedCallStatefulPartitionedCall#reshape_96/PartitionedCall:output:0conv1d_95_542341conv1d_95_542343*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:џџџџџџџџџt*$
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
E__inference_conv1d_95_layer_call_and_return_conditional_losses_542198Ћ
!dense_116/StatefulPartitionedCallStatefulPartitionedCall*conv1d_95/StatefulPartitionedCall:output:0dense_116_542346dense_116_542348*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:џџџџџџџџџX*$
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
E__inference_dense_116_layer_call_and_return_conditional_losses_542243
!max_pooling1d_115/PartitionedCallPartitionedCall*dense_116/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:џџџџџџџџџX* 
_read_only_resource_inputs
 *<
config_proto,*

CPU

GPU(2*0J

  zE8 *V
fQRO
M__inference_max_pooling1d_115_layer_call_and_return_conditional_losses_542169Ћ
!dense_117/StatefulPartitionedCallStatefulPartitionedCall*max_pooling1d_115/PartitionedCall:output:0dense_117_542352dense_117_542354*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:џџџџџџџџџ|*$
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
E__inference_dense_117_layer_call_and_return_conditional_losses_542281Ћ
!conv1d_96/StatefulPartitionedCallStatefulPartitionedCall*dense_117/StatefulPartitionedCall:output:0conv1d_96_542357conv1d_96_542359*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:џџџџџџџџџf*$
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
E__inference_conv1d_96_layer_call_and_return_conditional_losses_542303я
flatten_96/PartitionedCallPartitionedCall*conv1d_96/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџf* 
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
F__inference_flatten_96_layer_call_and_return_conditional_losses_542315И
'INJECTION_MASKS/StatefulPartitionedCallStatefulPartitionedCall#flatten_96/PartitionedCall:output:0injection_masks_542363injection_masks_542365*
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
K__inference_INJECTION_MASKS_layer_call_and_return_conditional_losses_542328
IdentityIdentity0INJECTION_MASKS/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ
NoOpNoOp(^INJECTION_MASKS/StatefulPartitionedCall"^conv1d_95/StatefulPartitionedCall"^conv1d_96/StatefulPartitionedCall"^dense_116/StatefulPartitionedCall"^dense_117/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*X
_input_shapesG
E:џџџџџџџџџ:џџџџџџџџџ : : : : : : : : : : 2R
'INJECTION_MASKS/StatefulPartitionedCall'INJECTION_MASKS/StatefulPartitionedCall2F
!conv1d_95/StatefulPartitionedCall!conv1d_95/StatefulPartitionedCall2F
!conv1d_96/StatefulPartitionedCall!conv1d_96/StatefulPartitionedCall2F
!dense_116/StatefulPartitionedCall!dense_116/StatefulPartitionedCall2F
!dense_117/StatefulPartitionedCall!dense_117/StatefulPartitionedCall:VR
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
ё

)__inference_model_96_layer_call_fn_542654
inputs_offsource
inputs_onsource
unknown:[t
	unknown_0:t
	unknown_1:tX
	unknown_2:X
	unknown_3:X|
	unknown_4:|
	unknown_5:<|f
	unknown_6:f
	unknown_7:f
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
D__inference_model_96_layer_call_and_return_conditional_losses_542407o
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


#__inference_internal_grad_fn_543178
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
:џџџџџџџџџXQ
SigmoidSigmoidmul:z:0*
T0*+
_output_shapes
:џџџџџџџџџXY
mul_1Mulmul_betamul_biasadd*
T0*+
_output_shapes
:џџџџџџџџџXJ
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?]
subSubsub/x:output:0Sigmoid:y:0*
T0*+
_output_shapes
:џџџџџџџџџXV
mul_2Mul	mul_1:z:0sub:z:0*
T0*+
_output_shapes
:џџџџџџџџџXJ
add/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?]
addAddV2add/x:output:0	mul_2:z:0*
T0*+
_output_shapes
:џџџџџџџџџXX
mul_3MulSigmoid:y:0add:z:0*
T0*+
_output_shapes
:џџџџџџџџџXS
SquareSquaremul_biasadd*
T0*+
_output_shapes
:џџџџџџџџџX^
mul_4Mulresult_grads_0
Square:y:0*
T0*+
_output_shapes
:џџџџџџџџџXZ
mul_5Mul	mul_4:z:0Sigmoid:y:0*
T0*+
_output_shapes
:џџџџџџџџџXL
sub_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?a
sub_1Subsub_1/x:output:0Sigmoid:y:0*
T0*+
_output_shapes
:џџџџџџџџџXX
mul_6Mul	mul_5:z:0	sub_1:z:0*
T0*+
_output_shapes
:џџџџџџџџџXZ
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
:џџџџџџџџџXU
IdentityIdentity	mul_7:z:0*
T0*+
_output_shapes
:џџџџџџџџџXE

Identity_1IdentitySum:output:0*
T0*
_output_shapes
: "!

identity_1Identity_1:output:0"
identityIdentity:output:0*\
_input_shapesK
I:џџџџџџџџџX:џџџџџџџџџX: : :џџџџџџџџџX:1-
+
_output_shapes
:џџџџџџџџџX:
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
:џџџџџџџџџX
(
_user_specified_nameresult_grads_1: 
&
 _has_manual_control_dependencies(
+
_output_shapes
:џџџџџџџџџX
(
_user_specified_nameresult_grads_0


#__inference_internal_grad_fn_543206
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
:џџџџџџџџџXQ
SigmoidSigmoidmul:z:0*
T0*+
_output_shapes
:џџџџџџџџџXY
mul_1Mulmul_betamul_biasadd*
T0*+
_output_shapes
:џџџџџџџџџXJ
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?]
subSubsub/x:output:0Sigmoid:y:0*
T0*+
_output_shapes
:џџџџџџџџџXV
mul_2Mul	mul_1:z:0sub:z:0*
T0*+
_output_shapes
:џџџџџџџџџXJ
add/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?]
addAddV2add/x:output:0	mul_2:z:0*
T0*+
_output_shapes
:џџџџџџџџџXX
mul_3MulSigmoid:y:0add:z:0*
T0*+
_output_shapes
:џџџџџџџџџXS
SquareSquaremul_biasadd*
T0*+
_output_shapes
:џџџџџџџџџX^
mul_4Mulresult_grads_0
Square:y:0*
T0*+
_output_shapes
:џџџџџџџџџXZ
mul_5Mul	mul_4:z:0Sigmoid:y:0*
T0*+
_output_shapes
:џџџџџџџџџXL
sub_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?a
sub_1Subsub_1/x:output:0Sigmoid:y:0*
T0*+
_output_shapes
:џџџџџџџџџXX
mul_6Mul	mul_5:z:0	sub_1:z:0*
T0*+
_output_shapes
:џџџџџџџџџXZ
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
:џџџџџџџџџXU
IdentityIdentity	mul_7:z:0*
T0*+
_output_shapes
:џџџџџџџџџXE

Identity_1IdentitySum:output:0*
T0*
_output_shapes
: "!

identity_1Identity_1:output:0"
identityIdentity:output:0*\
_input_shapesK
I:џџџџџџџџџX:џџџџџџџџџX: : :џџџџџџџџџX:1-
+
_output_shapes
:џџџџџџџџџX:
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
:џџџџџџџџџX
(
_user_specified_nameresult_grads_1: 
&
 _has_manual_control_dependencies(
+
_output_shapes
:џџџџџџџџџX
(
_user_specified_nameresult_grads_0
Ќ
K
#__inference__update_step_xla_286019
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
Ф
S
#__inference__update_step_xla_285974
gradient
variable:[t*
_XlaMustCompile(*(
_construction_contextkEagerRuntime*#
_input_shapes
:[t: *
	_noinline(:($
"
_user_specified_name
variable:L H
"
_output_shapes
:[t
"
_user_specified_name
gradient
Ё

ќ
K__inference_INJECTION_MASKS_layer_call_and_return_conditional_losses_542328

inputs0
matmul_readvariableop_resource:f-
biasadd_readvariableop_resource:
identityЂBiasAdd/ReadVariableOpЂMatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:f*
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
:џџџџџџџџџf: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:џџџџџџџџџf
 
_user_specified_nameinputs
}
	
D__inference_model_96_layer_call_and_return_conditional_losses_542892
inputs_offsource
inputs_onsourceK
5conv1d_95_conv1d_expanddims_1_readvariableop_resource:[t7
)conv1d_95_biasadd_readvariableop_resource:t=
+dense_116_tensordot_readvariableop_resource:tX7
)dense_116_biasadd_readvariableop_resource:X=
+dense_117_tensordot_readvariableop_resource:X|7
)dense_117_biasadd_readvariableop_resource:|K
5conv1d_96_conv1d_expanddims_1_readvariableop_resource:<|f7
)conv1d_96_biasadd_readvariableop_resource:f@
.injection_masks_matmul_readvariableop_resource:f=
/injection_masks_biasadd_readvariableop_resource:
identityЂ&INJECTION_MASKS/BiasAdd/ReadVariableOpЂ%INJECTION_MASKS/MatMul/ReadVariableOpЂ conv1d_95/BiasAdd/ReadVariableOpЂ,conv1d_95/Conv1D/ExpandDims_1/ReadVariableOpЂ conv1d_96/BiasAdd/ReadVariableOpЂ,conv1d_96/Conv1D/ExpandDims_1/ReadVariableOpЂ dense_116/BiasAdd/ReadVariableOpЂ"dense_116/Tensordot/ReadVariableOpЂ dense_117/BiasAdd/ReadVariableOpЂ"dense_117/Tensordot/ReadVariableOpШ
%whiten_passthrough_50/PartitionedCallPartitionedCallinputs_offsource*
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
)__inference_restored_function_body_285129л
reshape_96/PartitionedCallPartitionedCall.whiten_passthrough_50/PartitionedCall:output:0*
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
)__inference_restored_function_body_285135j
conv1d_95/Conv1D/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
§џџџџџџџџГ
conv1d_95/Conv1D/ExpandDims
ExpandDims#reshape_96/PartitionedCall:output:0(conv1d_95/Conv1D/ExpandDims/dim:output:0*
T0*0
_output_shapes
:џџџџџџџџџІ
,conv1d_95/Conv1D/ExpandDims_1/ReadVariableOpReadVariableOp5conv1d_95_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:[t*
dtype0c
!conv1d_95/Conv1D/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : О
conv1d_95/Conv1D/ExpandDims_1
ExpandDims4conv1d_95/Conv1D/ExpandDims_1/ReadVariableOp:value:0*conv1d_95/Conv1D/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:[tЪ
conv1d_95/Conv1DConv2D$conv1d_95/Conv1D/ExpandDims:output:0&conv1d_95/Conv1D/ExpandDims_1:output:0*
T0*/
_output_shapes
:џџџџџџџџџt*
paddingSAME*
strides
m
conv1d_95/Conv1D/SqueezeSqueezeconv1d_95/Conv1D:output:0*
T0*+
_output_shapes
:џџџџџџџџџt*
squeeze_dims

§џџџџџџџџ
 conv1d_95/BiasAdd/ReadVariableOpReadVariableOp)conv1d_95_biasadd_readvariableop_resource*
_output_shapes
:t*
dtype0
conv1d_95/BiasAddBiasAdd!conv1d_95/Conv1D/Squeeze:output:0(conv1d_95/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:џџџџџџџџџtn
conv1d_95/SoftmaxSoftmaxconv1d_95/BiasAdd:output:0*
T0*+
_output_shapes
:џџџџџџџџџt
"dense_116/Tensordot/ReadVariableOpReadVariableOp+dense_116_tensordot_readvariableop_resource*
_output_shapes

:tX*
dtype0b
dense_116/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:i
dense_116/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       r
dense_116/Tensordot/ShapeShapeconv1d_95/Softmax:softmax:0*
T0*
_output_shapes
::эЯc
!dense_116/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : у
dense_116/Tensordot/GatherV2GatherV2"dense_116/Tensordot/Shape:output:0!dense_116/Tensordot/free:output:0*dense_116/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:e
#dense_116/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ч
dense_116/Tensordot/GatherV2_1GatherV2"dense_116/Tensordot/Shape:output:0!dense_116/Tensordot/axes:output:0,dense_116/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:c
dense_116/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 
dense_116/Tensordot/ProdProd%dense_116/Tensordot/GatherV2:output:0"dense_116/Tensordot/Const:output:0*
T0*
_output_shapes
: e
dense_116/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 
dense_116/Tensordot/Prod_1Prod'dense_116/Tensordot/GatherV2_1:output:0$dense_116/Tensordot/Const_1:output:0*
T0*
_output_shapes
: a
dense_116/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : Ф
dense_116/Tensordot/concatConcatV2!dense_116/Tensordot/free:output:0!dense_116/Tensordot/axes:output:0(dense_116/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:
dense_116/Tensordot/stackPack!dense_116/Tensordot/Prod:output:0#dense_116/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:Ђ
dense_116/Tensordot/transpose	Transposeconv1d_95/Softmax:softmax:0#dense_116/Tensordot/concat:output:0*
T0*+
_output_shapes
:џџџџџџџџџtЈ
dense_116/Tensordot/ReshapeReshape!dense_116/Tensordot/transpose:y:0"dense_116/Tensordot/stack:output:0*
T0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџЈ
dense_116/Tensordot/MatMulMatMul$dense_116/Tensordot/Reshape:output:0*dense_116/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџXe
dense_116/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:Xc
!dense_116/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : Я
dense_116/Tensordot/concat_1ConcatV2%dense_116/Tensordot/GatherV2:output:0$dense_116/Tensordot/Const_2:output:0*dense_116/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:Ё
dense_116/TensordotReshape$dense_116/Tensordot/MatMul:product:0%dense_116/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:џџџџџџџџџX
 dense_116/BiasAdd/ReadVariableOpReadVariableOp)dense_116_biasadd_readvariableop_resource*
_output_shapes
:X*
dtype0
dense_116/BiasAddBiasAdddense_116/Tensordot:output:0(dense_116/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:џџџџџџџџџXS
dense_116/betaConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
dense_116/mulMuldense_116/beta:output:0dense_116/BiasAdd:output:0*
T0*+
_output_shapes
:џџџџџџџџџXe
dense_116/SigmoidSigmoiddense_116/mul:z:0*
T0*+
_output_shapes
:џџџџџџџџџX
dense_116/mul_1Muldense_116/BiasAdd:output:0dense_116/Sigmoid:y:0*
T0*+
_output_shapes
:џџџџџџџџџXi
dense_116/IdentityIdentitydense_116/mul_1:z:0*
T0*+
_output_shapes
:џџџџџџџџџXь
dense_116/IdentityN	IdentityNdense_116/mul_1:z:0dense_116/BiasAdd:output:0dense_116/beta:output:0*
T
2*,
_gradient_op_typeCustomGradient-542831*D
_output_shapes2
0:џџџџџџџџџX:џџџџџџџџџX: b
 max_pooling1d_115/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :­
max_pooling1d_115/ExpandDims
ExpandDimsdense_116/IdentityN:output:0)max_pooling1d_115/ExpandDims/dim:output:0*
T0*/
_output_shapes
:џџџџџџџџџXЗ
max_pooling1d_115/MaxPoolMaxPool%max_pooling1d_115/ExpandDims:output:0*/
_output_shapes
:џџџџџџџџџX*
ksize
*
paddingSAME*
strides
	
max_pooling1d_115/SqueezeSqueeze"max_pooling1d_115/MaxPool:output:0*
T0*+
_output_shapes
:џџџџџџџџџX*
squeeze_dims

"dense_117/Tensordot/ReadVariableOpReadVariableOp+dense_117_tensordot_readvariableop_resource*
_output_shapes

:X|*
dtype0b
dense_117/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:i
dense_117/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       y
dense_117/Tensordot/ShapeShape"max_pooling1d_115/Squeeze:output:0*
T0*
_output_shapes
::эЯc
!dense_117/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : у
dense_117/Tensordot/GatherV2GatherV2"dense_117/Tensordot/Shape:output:0!dense_117/Tensordot/free:output:0*dense_117/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:e
#dense_117/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ч
dense_117/Tensordot/GatherV2_1GatherV2"dense_117/Tensordot/Shape:output:0!dense_117/Tensordot/axes:output:0,dense_117/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:c
dense_117/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 
dense_117/Tensordot/ProdProd%dense_117/Tensordot/GatherV2:output:0"dense_117/Tensordot/Const:output:0*
T0*
_output_shapes
: e
dense_117/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 
dense_117/Tensordot/Prod_1Prod'dense_117/Tensordot/GatherV2_1:output:0$dense_117/Tensordot/Const_1:output:0*
T0*
_output_shapes
: a
dense_117/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : Ф
dense_117/Tensordot/concatConcatV2!dense_117/Tensordot/free:output:0!dense_117/Tensordot/axes:output:0(dense_117/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:
dense_117/Tensordot/stackPack!dense_117/Tensordot/Prod:output:0#dense_117/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:Љ
dense_117/Tensordot/transpose	Transpose"max_pooling1d_115/Squeeze:output:0#dense_117/Tensordot/concat:output:0*
T0*+
_output_shapes
:џџџџџџџџџXЈ
dense_117/Tensordot/ReshapeReshape!dense_117/Tensordot/transpose:y:0"dense_117/Tensordot/stack:output:0*
T0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџЈ
dense_117/Tensordot/MatMulMatMul$dense_117/Tensordot/Reshape:output:0*dense_117/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ|e
dense_117/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:|c
!dense_117/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : Я
dense_117/Tensordot/concat_1ConcatV2%dense_117/Tensordot/GatherV2:output:0$dense_117/Tensordot/Const_2:output:0*dense_117/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:Ё
dense_117/TensordotReshape$dense_117/Tensordot/MatMul:product:0%dense_117/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:џџџџџџџџџ|
 dense_117/BiasAdd/ReadVariableOpReadVariableOp)dense_117_biasadd_readvariableop_resource*
_output_shapes
:|*
dtype0
dense_117/BiasAddBiasAdddense_117/Tensordot:output:0(dense_117/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:џџџџџџџџџ|h
dense_117/ReluReludense_117/BiasAdd:output:0*
T0*+
_output_shapes
:џџџџџџџџџ|j
conv1d_96/Conv1D/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
§џџџџџџџџЋ
conv1d_96/Conv1D/ExpandDims
ExpandDimsdense_117/Relu:activations:0(conv1d_96/Conv1D/ExpandDims/dim:output:0*
T0*/
_output_shapes
:џџџџџџџџџ|І
,conv1d_96/Conv1D/ExpandDims_1/ReadVariableOpReadVariableOp5conv1d_96_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:<|f*
dtype0c
!conv1d_96/Conv1D/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : О
conv1d_96/Conv1D/ExpandDims_1
ExpandDims4conv1d_96/Conv1D/ExpandDims_1/ReadVariableOp:value:0*conv1d_96/Conv1D/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:<|fЪ
conv1d_96/Conv1DConv2D$conv1d_96/Conv1D/ExpandDims:output:0&conv1d_96/Conv1D/ExpandDims_1:output:0*
T0*/
_output_shapes
:џџџџџџџџџf*
paddingSAME*
strides
B
conv1d_96/Conv1D/SqueezeSqueezeconv1d_96/Conv1D:output:0*
T0*+
_output_shapes
:џџџџџџџџџf*
squeeze_dims

§џџџџџџџџ
 conv1d_96/BiasAdd/ReadVariableOpReadVariableOp)conv1d_96_biasadd_readvariableop_resource*
_output_shapes
:f*
dtype0
conv1d_96/BiasAddBiasAdd!conv1d_96/Conv1D/Squeeze:output:0(conv1d_96/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:џџџџџџџџџff
conv1d_96/EluEluconv1d_96/BiasAdd:output:0*
T0*+
_output_shapes
:џџџџџџџџџfa
flatten_96/ConstConst*
_output_shapes
:*
dtype0*
valueB"џџџџf   
flatten_96/ReshapeReshapeconv1d_96/Elu:activations:0flatten_96/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџf
%INJECTION_MASKS/MatMul/ReadVariableOpReadVariableOp.injection_masks_matmul_readvariableop_resource*
_output_shapes

:f*
dtype0
INJECTION_MASKS/MatMulMatMulflatten_96/Reshape:output:0-INJECTION_MASKS/MatMul/ReadVariableOp:value:0*
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
:џџџџџџџџџЫ
NoOpNoOp'^INJECTION_MASKS/BiasAdd/ReadVariableOp&^INJECTION_MASKS/MatMul/ReadVariableOp!^conv1d_95/BiasAdd/ReadVariableOp-^conv1d_95/Conv1D/ExpandDims_1/ReadVariableOp!^conv1d_96/BiasAdd/ReadVariableOp-^conv1d_96/Conv1D/ExpandDims_1/ReadVariableOp!^dense_116/BiasAdd/ReadVariableOp#^dense_116/Tensordot/ReadVariableOp!^dense_117/BiasAdd/ReadVariableOp#^dense_117/Tensordot/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*X
_input_shapesG
E:џџџџџџџџџ:џџџџџџџџџ : : : : : : : : : : 2P
&INJECTION_MASKS/BiasAdd/ReadVariableOp&INJECTION_MASKS/BiasAdd/ReadVariableOp2N
%INJECTION_MASKS/MatMul/ReadVariableOp%INJECTION_MASKS/MatMul/ReadVariableOp2D
 conv1d_95/BiasAdd/ReadVariableOp conv1d_95/BiasAdd/ReadVariableOp2\
,conv1d_95/Conv1D/ExpandDims_1/ReadVariableOp,conv1d_95/Conv1D/ExpandDims_1/ReadVariableOp2D
 conv1d_96/BiasAdd/ReadVariableOp conv1d_96/BiasAdd/ReadVariableOp2\
,conv1d_96/Conv1D/ExpandDims_1/ReadVariableOp,conv1d_96/Conv1D/ExpandDims_1/ReadVariableOp2D
 dense_116/BiasAdd/ReadVariableOp dense_116/BiasAdd/ReadVariableOp2H
"dense_116/Tensordot/ReadVariableOp"dense_116/Tensordot/ReadVariableOp2D
 dense_117/BiasAdd/ReadVariableOp dense_117/BiasAdd/ReadVariableOp2H
"dense_117/Tensordot/ReadVariableOp"dense_117/Tensordot/ReadVariableOp:]Y
,
_output_shapes
:џџџџџџџџџ 
)
_user_specified_nameinputs_onsource:_ [
-
_output_shapes
:џџџџџџџџџ
*
_user_specified_nameinputs_offsource<
#__inference_internal_grad_fn_543178CustomGradient-542956<
#__inference_internal_grad_fn_543206CustomGradient-542234<
#__inference_internal_grad_fn_543234CustomGradient-542831<
#__inference_internal_grad_fn_543262CustomGradient-542725<
#__inference_internal_grad_fn_543290CustomGradient-542099"ѓ
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
StatefulPartitionedCall:0џџџџџџџџџtensorflow/serving/predict:Ю

layer-0
layer-1
layer-2
layer_with_weights-0
layer-3
layer_with_weights-1
layer-4
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
р
/	variables
0trainable_variables
1regularization_losses
2	keras_api
3__call__
*4&call_and_return_all_conditional_losses

5kernel
6bias
#7_self_saveable_object_factories"
_tf_keras_layer
Ъ
8	variables
9trainable_variables
:regularization_losses
;	keras_api
<__call__
*=&call_and_return_all_conditional_losses
#>_self_saveable_object_factories"
_tf_keras_layer
р
?	variables
@trainable_variables
Aregularization_losses
B	keras_api
C__call__
*D&call_and_return_all_conditional_losses

Ekernel
Fbias
#G_self_saveable_object_factories"
_tf_keras_layer

H	variables
Itrainable_variables
Jregularization_losses
K	keras_api
L__call__
*M&call_and_return_all_conditional_losses

Nkernel
Obias
#P_self_saveable_object_factories
 Q_jit_compiled_convolution_op"
_tf_keras_layer
Ъ
R	variables
Strainable_variables
Tregularization_losses
U	keras_api
V__call__
*W&call_and_return_all_conditional_losses
#X_self_saveable_object_factories"
_tf_keras_layer
D
#Y_self_saveable_object_factories"
_tf_keras_input_layer
р
Z	variables
[trainable_variables
\regularization_losses
]	keras_api
^__call__
*_&call_and_return_all_conditional_losses

`kernel
abias
#b_self_saveable_object_factories"
_tf_keras_layer
f
+0
,1
52
63
E4
F5
N6
O7
`8
a9"
trackable_list_wrapper
f
+0
,1
52
63
E4
F5
N6
O7
`8
a9"
trackable_list_wrapper
 "
trackable_list_wrapper
Ъ
cnon_trainable_variables

dlayers
emetrics
flayer_regularization_losses
glayer_metrics
	variables
trainable_variables
regularization_losses
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
Я
htrace_0
itrace_1
jtrace_2
ktrace_32ф
)__inference_model_96_layer_call_fn_542430
)__inference_model_96_layer_call_fn_542490
)__inference_model_96_layer_call_fn_542654
)__inference_model_96_layer_call_fn_542680Е
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
 zhtrace_0zitrace_1zjtrace_2zktrace_3
Л
ltrace_0
mtrace_1
ntrace_2
otrace_32а
D__inference_model_96_layer_call_and_return_conditional_losses_542335
D__inference_model_96_layer_call_and_return_conditional_losses_542369
D__inference_model_96_layer_call_and_return_conditional_losses_542786
D__inference_model_96_layer_call_and_return_conditional_losses_542892Е
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
 zltrace_0zmtrace_1zntrace_2zotrace_3
иBе
!__inference__wrapped_model_542160	OFFSOURCEONSOURCE"
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
p
_variables
q_iterations
r_learning_rate
s_index_dict
t
_momentums
u_velocities
v_update_step_xla"
experimentalOptimizer
,
wserving_default"
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
xnon_trainable_variables

ylayers
zmetrics
{layer_regularization_losses
|layer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
ю
}trace_02б
4__inference_whiten_passthrough_50_layer_call_fn_1101
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

~trace_02ы
N__inference_whiten_passthrough_50_layer_call_and_return_conditional_losses_833
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
 z~trace_0
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
Б
non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
	variables
trainable_variables
 regularization_losses
"__call__
*#&call_and_return_all_conditional_losses
&#"call_and_return_conditional_losses"
_generic_user_object
х
trace_02Ц
)__inference_reshape_96_layer_call_fn_1449
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
џ
trace_02р
C__inference_reshape_96_layer_call_and_return_conditional_losses_309
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
 ztrace_0
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
non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
%	variables
&trainable_variables
'regularization_losses
)__call__
**&call_and_return_all_conditional_losses
&*"call_and_return_conditional_losses"
_generic_user_object
ц
trace_02Ч
*__inference_conv1d_95_layer_call_fn_542901
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

trace_02т
E__inference_conv1d_95_layer_call_and_return_conditional_losses_542917
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
 ztrace_0
:[t 2kernel
:t 2bias
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
50
61"
trackable_list_wrapper
.
50
61"
trackable_list_wrapper
 "
trackable_list_wrapper
В
non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
/	variables
0trainable_variables
1regularization_losses
3__call__
*4&call_and_return_all_conditional_losses
&4"call_and_return_conditional_losses"
_generic_user_object
ц
trace_02Ч
*__inference_dense_116_layer_call_fn_542926
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

trace_02т
E__inference_dense_116_layer_call_and_return_conditional_losses_542965
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
 ztrace_0
:tX 2kernel
:X 2bias
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
В
non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
8	variables
9trainable_variables
:regularization_losses
<__call__
*=&call_and_return_all_conditional_losses
&="call_and_return_conditional_losses"
_generic_user_object
ю
trace_02Я
2__inference_max_pooling1d_115_layer_call_fn_542970
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

trace_02ъ
M__inference_max_pooling1d_115_layer_call_and_return_conditional_losses_542978
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
 ztrace_0
 "
trackable_dict_wrapper
.
E0
F1"
trackable_list_wrapper
.
E0
F1"
trackable_list_wrapper
 "
trackable_list_wrapper
В
non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
?	variables
@trainable_variables
Aregularization_losses
C__call__
*D&call_and_return_all_conditional_losses
&D"call_and_return_conditional_losses"
_generic_user_object
ц
 trace_02Ч
*__inference_dense_117_layer_call_fn_542987
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

Ёtrace_02т
E__inference_dense_117_layer_call_and_return_conditional_losses_543018
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
 zЁtrace_0
:X| 2kernel
:| 2bias
 "
trackable_dict_wrapper
.
N0
O1"
trackable_list_wrapper
.
N0
O1"
trackable_list_wrapper
 "
trackable_list_wrapper
В
Ђnon_trainable_variables
Ѓlayers
Єmetrics
 Ѕlayer_regularization_losses
Іlayer_metrics
H	variables
Itrainable_variables
Jregularization_losses
L__call__
*M&call_and_return_all_conditional_losses
&M"call_and_return_conditional_losses"
_generic_user_object
ц
Їtrace_02Ч
*__inference_conv1d_96_layer_call_fn_543027
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
 zЇtrace_0

Јtrace_02т
E__inference_conv1d_96_layer_call_and_return_conditional_losses_543043
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
 zЈtrace_0
:<|f 2kernel
:f 2bias
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
Љnon_trainable_variables
Њlayers
Ћmetrics
 Ќlayer_regularization_losses
­layer_metrics
R	variables
Strainable_variables
Tregularization_losses
V__call__
*W&call_and_return_all_conditional_losses
&W"call_and_return_conditional_losses"
_generic_user_object
ч
Ўtrace_02Ш
+__inference_flatten_96_layer_call_fn_543048
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
 zЎtrace_0

Џtrace_02у
F__inference_flatten_96_layer_call_and_return_conditional_losses_543054
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
 zЏtrace_0
 "
trackable_dict_wrapper
 "
trackable_dict_wrapper
.
`0
a1"
trackable_list_wrapper
.
`0
a1"
trackable_list_wrapper
 "
trackable_list_wrapper
В
Аnon_trainable_variables
Бlayers
Вmetrics
 Гlayer_regularization_losses
Дlayer_metrics
Z	variables
[trainable_variables
\regularization_losses
^__call__
*_&call_and_return_all_conditional_losses
&_"call_and_return_conditional_losses"
_generic_user_object
ь
Еtrace_02Э
0__inference_INJECTION_MASKS_layer_call_fn_543063
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
 zЕtrace_0

Жtrace_02ш
K__inference_INJECTION_MASKS_layer_call_and_return_conditional_losses_543074
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
 zЖtrace_0
:f 2kernel
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
З0
И1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
§Bњ
)__inference_model_96_layer_call_fn_542430	OFFSOURCEONSOURCE"Е
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
)__inference_model_96_layer_call_fn_542490	OFFSOURCEONSOURCE"Е
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
)__inference_model_96_layer_call_fn_542654inputs_offsourceinputs_onsource"Е
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
)__inference_model_96_layer_call_fn_542680inputs_offsourceinputs_onsource"Е
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
D__inference_model_96_layer_call_and_return_conditional_losses_542335	OFFSOURCEONSOURCE"Е
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
D__inference_model_96_layer_call_and_return_conditional_losses_542369	OFFSOURCEONSOURCE"Е
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
D__inference_model_96_layer_call_and_return_conditional_losses_542786inputs_offsourceinputs_onsource"Е
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
D__inference_model_96_layer_call_and_return_conditional_losses_542892inputs_offsourceinputs_onsource"Е
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
q0
Й1
К2
Л3
М4
Н5
О6
П7
Р8
С9
Т10
У11
Ф12
Х13
Ц14
Ч15
Ш16
Щ17
Ъ18
Ы19
Ь20"
trackable_list_wrapper
:	  2	iteration
:  2learning_rate
 "
trackable_dict_wrapper
p
Й0
Л1
Н2
П3
С4
У5
Х6
Ч7
Щ8
Ы9"
trackable_list_wrapper
p
К0
М1
О2
Р3
Т4
Ф5
Ц6
Ш7
Ъ8
Ь9"
trackable_list_wrapper
П
Эtrace_0
Юtrace_1
Яtrace_2
аtrace_3
бtrace_4
вtrace_5
гtrace_6
дtrace_7
еtrace_8
жtrace_92Є
#__inference__update_step_xla_285974
#__inference__update_step_xla_285979
#__inference__update_step_xla_285984
#__inference__update_step_xla_285989
#__inference__update_step_xla_285994
#__inference__update_step_xla_285999
#__inference__update_step_xla_286004
#__inference__update_step_xla_286009
#__inference__update_step_xla_286014
#__inference__update_step_xla_286019Џ
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
 0zЭtrace_0zЮtrace_1zЯtrace_2zаtrace_3zбtrace_4zвtrace_5zгtrace_6zдtrace_7zеtrace_8zжtrace_9
еBв
$__inference_signature_wrapper_542628	OFFSOURCEONSOURCE"
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
4__inference_whiten_passthrough_50_layer_call_fn_1101inputs"
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
N__inference_whiten_passthrough_50_layer_call_and_return_conditional_losses_833inputs"
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
гBа
)__inference_reshape_96_layer_call_fn_1449inputs"
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
C__inference_reshape_96_layer_call_and_return_conditional_losses_309inputs"
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
*__inference_conv1d_95_layer_call_fn_542901inputs"
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
E__inference_conv1d_95_layer_call_and_return_conditional_losses_542917inputs"
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
*__inference_dense_116_layer_call_fn_542926inputs"
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
E__inference_dense_116_layer_call_and_return_conditional_losses_542965inputs"
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
мBй
2__inference_max_pooling1d_115_layer_call_fn_542970inputs"
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
їBє
M__inference_max_pooling1d_115_layer_call_and_return_conditional_losses_542978inputs"
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
*__inference_dense_117_layer_call_fn_542987inputs"
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
E__inference_dense_117_layer_call_and_return_conditional_losses_543018inputs"
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
*__inference_conv1d_96_layer_call_fn_543027inputs"
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
E__inference_conv1d_96_layer_call_and_return_conditional_losses_543043inputs"
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
+__inference_flatten_96_layer_call_fn_543048inputs"
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
F__inference_flatten_96_layer_call_and_return_conditional_losses_543054inputs"
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
0__inference_INJECTION_MASKS_layer_call_fn_543063inputs"
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
K__inference_INJECTION_MASKS_layer_call_and_return_conditional_losses_543074inputs"
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
з	variables
и	keras_api

йtotal

кcount"
_tf_keras_metric
c
л	variables
м	keras_api

нtotal

оcount
п
_fn_kwargs"
_tf_keras_metric
#:![t 2Adam/m/kernel
#:![t 2Adam/v/kernel
:t 2Adam/m/bias
:t 2Adam/v/bias
:tX 2Adam/m/kernel
:tX 2Adam/v/kernel
:X 2Adam/m/bias
:X 2Adam/v/bias
:X| 2Adam/m/kernel
:X| 2Adam/v/kernel
:| 2Adam/m/bias
:| 2Adam/v/bias
#:!<|f 2Adam/m/kernel
#:!<|f 2Adam/v/kernel
:f 2Adam/m/bias
:f 2Adam/v/bias
:f 2Adam/m/kernel
:f 2Adam/v/kernel
: 2Adam/m/bias
: 2Adam/v/bias
юBы
#__inference__update_step_xla_285974gradientvariable"­
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
#__inference__update_step_xla_285979gradientvariable"­
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
#__inference__update_step_xla_285984gradientvariable"­
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
#__inference__update_step_xla_285989gradientvariable"­
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
#__inference__update_step_xla_285994gradientvariable"­
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
#__inference__update_step_xla_285999gradientvariable"­
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
#__inference__update_step_xla_286004gradientvariable"­
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
#__inference__update_step_xla_286009gradientvariable"­
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
#__inference__update_step_xla_286014gradientvariable"­
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
#__inference__update_step_xla_286019gradientvariable"­
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
й0
к1"
trackable_list_wrapper
.
з	variables"
_generic_user_object
:  (2total
:  (2count
0
н0
о1"
trackable_list_wrapper
.
л	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapper
QbO
beta:0E__inference_dense_116_layer_call_and_return_conditional_losses_542965
TbR
	BiasAdd:0E__inference_dense_116_layer_call_and_return_conditional_losses_542965
QbO
beta:0E__inference_dense_116_layer_call_and_return_conditional_losses_542243
TbR
	BiasAdd:0E__inference_dense_116_layer_call_and_return_conditional_losses_542243
ZbX
dense_116/beta:0D__inference_model_96_layer_call_and_return_conditional_losses_542892
]b[
dense_116/BiasAdd:0D__inference_model_96_layer_call_and_return_conditional_losses_542892
ZbX
dense_116/beta:0D__inference_model_96_layer_call_and_return_conditional_losses_542786
]b[
dense_116/BiasAdd:0D__inference_model_96_layer_call_and_return_conditional_losses_542786
@b>
model_96/dense_116/beta:0!__inference__wrapped_model_542160
CbA
model_96/dense_116/BiasAdd:0!__inference__wrapped_model_542160В
K__inference_INJECTION_MASKS_layer_call_and_return_conditional_losses_543074c`a/Ђ,
%Ђ"
 
inputsџџџџџџџџџf
Њ ",Ђ)
"
tensor_0џџџџџџџџџ
 
0__inference_INJECTION_MASKS_layer_call_fn_543063X`a/Ђ,
%Ђ"
 
inputsџџџџџџџџџf
Њ "!
unknownџџџџџџџџџ
#__inference__update_step_xla_285974vpЂm
fЂc

gradient[t
85	!Ђ
њ[t

p
` VariableSpec 
`реп?
Њ "
 
#__inference__update_step_xla_285979f`Ђ]
VЂS

gradientt
0-	Ђ
њt

p
` VariableSpec 
`рѕдп?
Њ "
 
#__inference__update_step_xla_285984nhЂe
^Ђ[

gradienttX
41	Ђ
њtX

p
` VariableSpec 
`рбЪп?
Њ "
 
#__inference__update_step_xla_285989f`Ђ]
VЂS

gradientX
0-	Ђ
њX

p
` VariableSpec 
`рЪп?
Њ "
 
#__inference__update_step_xla_285994nhЂe
^Ђ[

gradientX|
41	Ђ
њX|

p
` VariableSpec 
`р­Пп?
Њ "
 
#__inference__update_step_xla_285999f`Ђ]
VЂS

gradient|
0-	Ђ
њ|

p
` VariableSpec 
`рїОп?
Њ "
 
#__inference__update_step_xla_286004vpЂm
fЂc

gradient<|f
85	!Ђ
њ<|f

p
` VariableSpec 
`рУФп?
Њ "
 
#__inference__update_step_xla_286009f`Ђ]
VЂS

gradientf
0-	Ђ
њf

p
` VariableSpec 
`реФп?
Њ "
 
#__inference__update_step_xla_286014nhЂe
^Ђ[

gradientf
41	Ђ
њf

p
` VariableSpec 
`руФп?
Њ "
 
#__inference__update_step_xla_286019f`Ђ]
VЂS

gradient
0-	Ђ
њ

p
` VariableSpec 
`рЬСп?
Њ "
 і
!__inference__wrapped_model_542160а
+,56EFNO`aЂ|
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
E__inference_conv1d_95_layer_call_and_return_conditional_losses_542917l+,4Ђ1
*Ђ'
%"
inputsџџџџџџџџџ
Њ "0Ђ-
&#
tensor_0џџџџџџџџџt
 
*__inference_conv1d_95_layer_call_fn_542901a+,4Ђ1
*Ђ'
%"
inputsџџџџџџџџџ
Њ "%"
unknownџџџџџџџџџtД
E__inference_conv1d_96_layer_call_and_return_conditional_losses_543043kNO3Ђ0
)Ђ&
$!
inputsџџџџџџџџџ|
Њ "0Ђ-
&#
tensor_0џџџџџџџџџf
 
*__inference_conv1d_96_layer_call_fn_543027`NO3Ђ0
)Ђ&
$!
inputsџџџџџџџџџ|
Њ "%"
unknownџџџџџџџџџfД
E__inference_dense_116_layer_call_and_return_conditional_losses_542965k563Ђ0
)Ђ&
$!
inputsџџџџџџџџџt
Њ "0Ђ-
&#
tensor_0џџџџџџџџџX
 
*__inference_dense_116_layer_call_fn_542926`563Ђ0
)Ђ&
$!
inputsџџџџџџџџџt
Њ "%"
unknownџџџџџџџџџXД
E__inference_dense_117_layer_call_and_return_conditional_losses_543018kEF3Ђ0
)Ђ&
$!
inputsџџџџџџџџџX
Њ "0Ђ-
&#
tensor_0џџџџџџџџџ|
 
*__inference_dense_117_layer_call_fn_542987`EF3Ђ0
)Ђ&
$!
inputsџџџџџџџџџX
Њ "%"
unknownџџџџџџџџџ|­
F__inference_flatten_96_layer_call_and_return_conditional_losses_543054c3Ђ0
)Ђ&
$!
inputsџџџџџџџџџf
Њ ",Ђ)
"
tensor_0џџџџџџџџџf
 
+__inference_flatten_96_layer_call_fn_543048X3Ђ0
)Ђ&
$!
inputsџџџџџџџџџf
Њ "!
unknownџџџџџџџџџfќ
#__inference_internal_grad_fn_543178дрсЂ
|Ђy

 
,)
result_grads_0џџџџџџџџџX
,)
result_grads_1џџџџџџџџџX

result_grads_2 
Њ "B?

 
&#
tensor_1џџџџџџџџџX

tensor_2 ќ
#__inference_internal_grad_fn_543206дтуЂ
|Ђy

 
,)
result_grads_0џџџџџџџџџX
,)
result_grads_1џџџџџџџџџX

result_grads_2 
Њ "B?

 
&#
tensor_1џџџџџџџџџX

tensor_2 ќ
#__inference_internal_grad_fn_543234дфхЂ
|Ђy

 
,)
result_grads_0џџџџџџџџџX
,)
result_grads_1џџџџџџџџџX

result_grads_2 
Њ "B?

 
&#
tensor_1џџџџџџџџџX

tensor_2 ќ
#__inference_internal_grad_fn_543262дцчЂ
|Ђy

 
,)
result_grads_0џџџџџџџџџX
,)
result_grads_1џџџџџџџџџX

result_grads_2 
Њ "B?

 
&#
tensor_1џџџџџџџџџX

tensor_2 ќ
#__inference_internal_grad_fn_543290дшщЂ
|Ђy

 
,)
result_grads_0џџџџџџџџџX
,)
result_grads_1џџџџџџџџџX

result_grads_2 
Њ "B?

 
&#
tensor_1џџџџџџџџџX

tensor_2 н
M__inference_max_pooling1d_115_layer_call_and_return_conditional_losses_542978EЂB
;Ђ8
63
inputs'џџџџџџџџџџџџџџџџџџџџџџџџџџџ
Њ "BЂ?
85
tensor_0'џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 З
2__inference_max_pooling1d_115_layer_call_fn_542970EЂB
;Ђ8
63
inputs'џџџџџџџџџџџџџџџџџџџџџџџџџџџ
Њ "74
unknown'џџџџџџџџџџџџџџџџџџџџџџџџџџџ
D__inference_model_96_layer_call_and_return_conditional_losses_542335Х
+,56EFNO`aЂ
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
D__inference_model_96_layer_call_and_return_conditional_losses_542369Х
+,56EFNO`aЂ
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
D__inference_model_96_layer_call_and_return_conditional_losses_542786е
+,56EFNO`aЂ
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
D__inference_model_96_layer_call_and_return_conditional_losses_542892е
+,56EFNO`aЂ
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
)__inference_model_96_layer_call_fn_542430К
+,56EFNO`aЂ
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
)__inference_model_96_layer_call_fn_542490К
+,56EFNO`aЂ
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
)__inference_model_96_layer_call_fn_542654Ъ
+,56EFNO`aЂ
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
)__inference_model_96_layer_call_fn_542680Ъ
+,56EFNO`aЂ
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
C__inference_reshape_96_layer_call_and_return_conditional_losses_309i4Ђ1
*Ђ'
%"
inputsџџџџџџџџџ
Њ "1Ђ.
'$
tensor_0џџџџџџџџџ
 
)__inference_reshape_96_layer_call_fn_1449^4Ђ1
*Ђ'
%"
inputsџџџџџџџџџ
Њ "&#
unknownџџџџџџџџџє
$__inference_signature_wrapper_542628Ы
+,56EFNO`azЂw
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
N__inference_whiten_passthrough_50_layer_call_and_return_conditional_losses_833j5Ђ2
+Ђ(
&#
inputsџџџџџџџџџ
Њ "1Ђ.
'$
tensor_0џџџџџџџџџ
 
4__inference_whiten_passthrough_50_layer_call_fn_1101_5Ђ2
+Ђ(
&#
inputsџџџџџџџџџ
Њ "&#
unknownџџџџџџџџџ