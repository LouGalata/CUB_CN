�#
��
8
Const
output"dtype"
valuetensor"
dtypetype

NoOp
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetype�
�
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
executor_typestring �
q
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape�"serve*2.1.02unknown8��
�
input_conv_21/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameinput_conv_21/kernel
�
(input_conv_21/kernel/Read/ReadVariableOpReadVariableOpinput_conv_21/kernel*&
_output_shapes
:*
dtype0
|
input_conv_21/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*#
shared_nameinput_conv_21/bias
u
&input_conv_21/bias/Read/ReadVariableOpReadVariableOpinput_conv_21/bias*
_output_shapes
:*
dtype0
�
"input_batch_normalization_21/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*3
shared_name$"input_batch_normalization_21/gamma
�
6input_batch_normalization_21/gamma/Read/ReadVariableOpReadVariableOp"input_batch_normalization_21/gamma*
_output_shapes
:*
dtype0
�
!input_batch_normalization_21/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*2
shared_name#!input_batch_normalization_21/beta
�
5input_batch_normalization_21/beta/Read/ReadVariableOpReadVariableOp!input_batch_normalization_21/beta*
_output_shapes
:*
dtype0
�
(input_batch_normalization_21/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:*9
shared_name*(input_batch_normalization_21/moving_mean
�
<input_batch_normalization_21/moving_mean/Read/ReadVariableOpReadVariableOp(input_batch_normalization_21/moving_mean*
_output_shapes
:*
dtype0
�
,input_batch_normalization_21/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:*=
shared_name.,input_batch_normalization_21/moving_variance
�
@input_batch_normalization_21/moving_variance/Read/ReadVariableOpReadVariableOp,input_batch_normalization_21/moving_variance*
_output_shapes
:*
dtype0
�
conv_1_21/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*!
shared_nameconv_1_21/kernel
}
$conv_1_21/kernel/Read/ReadVariableOpReadVariableOpconv_1_21/kernel*&
_output_shapes
:*
dtype0
t
conv_1_21/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameconv_1_21/bias
m
"conv_1_21/bias/Read/ReadVariableOpReadVariableOpconv_1_21/bias*
_output_shapes
:*
dtype0
z
b_norm_1_21/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*"
shared_nameb_norm_1_21/gamma
s
%b_norm_1_21/gamma/Read/ReadVariableOpReadVariableOpb_norm_1_21/gamma*
_output_shapes
:*
dtype0
x
b_norm_1_21/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*!
shared_nameb_norm_1_21/beta
q
$b_norm_1_21/beta/Read/ReadVariableOpReadVariableOpb_norm_1_21/beta*
_output_shapes
:*
dtype0
�
b_norm_1_21/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:*(
shared_nameb_norm_1_21/moving_mean

+b_norm_1_21/moving_mean/Read/ReadVariableOpReadVariableOpb_norm_1_21/moving_mean*
_output_shapes
:*
dtype0
�
b_norm_1_21/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:*,
shared_nameb_norm_1_21/moving_variance
�
/b_norm_1_21/moving_variance/Read/ReadVariableOpReadVariableOpb_norm_1_21/moving_variance*
_output_shapes
:*
dtype0
�
conv_2_21/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: *!
shared_nameconv_2_21/kernel
}
$conv_2_21/kernel/Read/ReadVariableOpReadVariableOpconv_2_21/kernel*&
_output_shapes
: *
dtype0
t
conv_2_21/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameconv_2_21/bias
m
"conv_2_21/bias/Read/ReadVariableOpReadVariableOpconv_2_21/bias*
_output_shapes
: *
dtype0
z
b_norm_2_21/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape: *"
shared_nameb_norm_2_21/gamma
s
%b_norm_2_21/gamma/Read/ReadVariableOpReadVariableOpb_norm_2_21/gamma*
_output_shapes
: *
dtype0
x
b_norm_2_21/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape: *!
shared_nameb_norm_2_21/beta
q
$b_norm_2_21/beta/Read/ReadVariableOpReadVariableOpb_norm_2_21/beta*
_output_shapes
: *
dtype0
�
b_norm_2_21/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape: *(
shared_nameb_norm_2_21/moving_mean

+b_norm_2_21/moving_mean/Read/ReadVariableOpReadVariableOpb_norm_2_21/moving_mean*
_output_shapes
: *
dtype0
�
b_norm_2_21/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape: *,
shared_nameb_norm_2_21/moving_variance
�
/b_norm_2_21/moving_variance/Read/ReadVariableOpReadVariableOpb_norm_2_21/moving_variance*
_output_shapes
: *
dtype0
�
conv_3_13/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: @*!
shared_nameconv_3_13/kernel
}
$conv_3_13/kernel/Read/ReadVariableOpReadVariableOpconv_3_13/kernel*&
_output_shapes
: @*
dtype0
t
conv_3_13/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_nameconv_3_13/bias
m
"conv_3_13/bias/Read/ReadVariableOpReadVariableOpconv_3_13/bias*
_output_shapes
:@*
dtype0
z
b_norm_3_13/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*"
shared_nameb_norm_3_13/gamma
s
%b_norm_3_13/gamma/Read/ReadVariableOpReadVariableOpb_norm_3_13/gamma*
_output_shapes
:@*
dtype0
x
b_norm_3_13/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*!
shared_nameb_norm_3_13/beta
q
$b_norm_3_13/beta/Read/ReadVariableOpReadVariableOpb_norm_3_13/beta*
_output_shapes
:@*
dtype0
�
b_norm_3_13/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*(
shared_nameb_norm_3_13/moving_mean

+b_norm_3_13/moving_mean/Read/ReadVariableOpReadVariableOpb_norm_3_13/moving_mean*
_output_shapes
:@*
dtype0
�
b_norm_3_13/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*,
shared_nameb_norm_3_13/moving_variance
�
/b_norm_3_13/moving_variance/Read/ReadVariableOpReadVariableOpb_norm_3_13/moving_variance*
_output_shapes
:@*
dtype0
�
conv_4_5/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@@* 
shared_nameconv_4_5/kernel
{
#conv_4_5/kernel/Read/ReadVariableOpReadVariableOpconv_4_5/kernel*&
_output_shapes
:@@*
dtype0
r
conv_4_5/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_nameconv_4_5/bias
k
!conv_4_5/bias/Read/ReadVariableOpReadVariableOpconv_4_5/bias*
_output_shapes
:@*
dtype0
x
b_norm_4_5/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*!
shared_nameb_norm_4_5/gamma
q
$b_norm_4_5/gamma/Read/ReadVariableOpReadVariableOpb_norm_4_5/gamma*
_output_shapes
:@*
dtype0
v
b_norm_4_5/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:@* 
shared_nameb_norm_4_5/beta
o
#b_norm_4_5/beta/Read/ReadVariableOpReadVariableOpb_norm_4_5/beta*
_output_shapes
:@*
dtype0
�
b_norm_4_5/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*'
shared_nameb_norm_4_5/moving_mean
}
*b_norm_4_5/moving_mean/Read/ReadVariableOpReadVariableOpb_norm_4_5/moving_mean*
_output_shapes
:@*
dtype0
�
b_norm_4_5/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*+
shared_nameb_norm_4_5/moving_variance
�
.b_norm_4_5/moving_variance/Read/ReadVariableOpReadVariableOpb_norm_4_5/moving_variance*
_output_shapes
:@*
dtype0
�
#output_class_distribution_21/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
��*4
shared_name%#output_class_distribution_21/kernel
�
7output_class_distribution_21/kernel/Read/ReadVariableOpReadVariableOp#output_class_distribution_21/kernel* 
_output_shapes
:
��*
dtype0
�
!output_class_distribution_21/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*2
shared_name#!output_class_distribution_21/bias
�
5output_class_distribution_21/bias/Read/ReadVariableOpReadVariableOp!output_class_distribution_21/bias*
_output_shapes	
:�*
dtype0
f
	Adam/iterVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_name	Adam/iter
_
Adam/iter/Read/ReadVariableOpReadVariableOp	Adam/iter*
_output_shapes
: *
dtype0	
j
Adam/beta_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_1
c
Adam/beta_1/Read/ReadVariableOpReadVariableOpAdam/beta_1*
_output_shapes
: *
dtype0
j
Adam/beta_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_2
c
Adam/beta_2/Read/ReadVariableOpReadVariableOpAdam/beta_2*
_output_shapes
: *
dtype0
h

Adam/decayVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name
Adam/decay
a
Adam/decay/Read/ReadVariableOpReadVariableOp
Adam/decay*
_output_shapes
: *
dtype0
x
Adam/learning_rateVarHandleOp*
_output_shapes
: *
dtype0*
shape: *#
shared_nameAdam/learning_rate
q
&Adam/learning_rate/Read/ReadVariableOpReadVariableOpAdam/learning_rate*
_output_shapes
: *
dtype0
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
�
Adam/input_conv_21/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*,
shared_nameAdam/input_conv_21/kernel/m
�
/Adam/input_conv_21/kernel/m/Read/ReadVariableOpReadVariableOpAdam/input_conv_21/kernel/m*&
_output_shapes
:*
dtype0
�
Adam/input_conv_21/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:**
shared_nameAdam/input_conv_21/bias/m
�
-Adam/input_conv_21/bias/m/Read/ReadVariableOpReadVariableOpAdam/input_conv_21/bias/m*
_output_shapes
:*
dtype0
�
)Adam/input_batch_normalization_21/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*:
shared_name+)Adam/input_batch_normalization_21/gamma/m
�
=Adam/input_batch_normalization_21/gamma/m/Read/ReadVariableOpReadVariableOp)Adam/input_batch_normalization_21/gamma/m*
_output_shapes
:*
dtype0
�
(Adam/input_batch_normalization_21/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*9
shared_name*(Adam/input_batch_normalization_21/beta/m
�
<Adam/input_batch_normalization_21/beta/m/Read/ReadVariableOpReadVariableOp(Adam/input_batch_normalization_21/beta/m*
_output_shapes
:*
dtype0
�
Adam/conv_1_21/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*(
shared_nameAdam/conv_1_21/kernel/m
�
+Adam/conv_1_21/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv_1_21/kernel/m*&
_output_shapes
:*
dtype0
�
Adam/conv_1_21/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*&
shared_nameAdam/conv_1_21/bias/m
{
)Adam/conv_1_21/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv_1_21/bias/m*
_output_shapes
:*
dtype0
�
Adam/b_norm_1_21/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*)
shared_nameAdam/b_norm_1_21/gamma/m
�
,Adam/b_norm_1_21/gamma/m/Read/ReadVariableOpReadVariableOpAdam/b_norm_1_21/gamma/m*
_output_shapes
:*
dtype0
�
Adam/b_norm_1_21/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*(
shared_nameAdam/b_norm_1_21/beta/m

+Adam/b_norm_1_21/beta/m/Read/ReadVariableOpReadVariableOpAdam/b_norm_1_21/beta/m*
_output_shapes
:*
dtype0
�
Adam/conv_2_21/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *(
shared_nameAdam/conv_2_21/kernel/m
�
+Adam/conv_2_21/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv_2_21/kernel/m*&
_output_shapes
: *
dtype0
�
Adam/conv_2_21/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *&
shared_nameAdam/conv_2_21/bias/m
{
)Adam/conv_2_21/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv_2_21/bias/m*
_output_shapes
: *
dtype0
�
Adam/b_norm_2_21/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *)
shared_nameAdam/b_norm_2_21/gamma/m
�
,Adam/b_norm_2_21/gamma/m/Read/ReadVariableOpReadVariableOpAdam/b_norm_2_21/gamma/m*
_output_shapes
: *
dtype0
�
Adam/b_norm_2_21/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *(
shared_nameAdam/b_norm_2_21/beta/m

+Adam/b_norm_2_21/beta/m/Read/ReadVariableOpReadVariableOpAdam/b_norm_2_21/beta/m*
_output_shapes
: *
dtype0
�
Adam/conv_3_13/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: @*(
shared_nameAdam/conv_3_13/kernel/m
�
+Adam/conv_3_13/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv_3_13/kernel/m*&
_output_shapes
: @*
dtype0
�
Adam/conv_3_13/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*&
shared_nameAdam/conv_3_13/bias/m
{
)Adam/conv_3_13/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv_3_13/bias/m*
_output_shapes
:@*
dtype0
�
Adam/b_norm_3_13/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*)
shared_nameAdam/b_norm_3_13/gamma/m
�
,Adam/b_norm_3_13/gamma/m/Read/ReadVariableOpReadVariableOpAdam/b_norm_3_13/gamma/m*
_output_shapes
:@*
dtype0
�
Adam/b_norm_3_13/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*(
shared_nameAdam/b_norm_3_13/beta/m

+Adam/b_norm_3_13/beta/m/Read/ReadVariableOpReadVariableOpAdam/b_norm_3_13/beta/m*
_output_shapes
:@*
dtype0
�
Adam/conv_4_5/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@@*'
shared_nameAdam/conv_4_5/kernel/m
�
*Adam/conv_4_5/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv_4_5/kernel/m*&
_output_shapes
:@@*
dtype0
�
Adam/conv_4_5/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*%
shared_nameAdam/conv_4_5/bias/m
y
(Adam/conv_4_5/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv_4_5/bias/m*
_output_shapes
:@*
dtype0
�
Adam/b_norm_4_5/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*(
shared_nameAdam/b_norm_4_5/gamma/m

+Adam/b_norm_4_5/gamma/m/Read/ReadVariableOpReadVariableOpAdam/b_norm_4_5/gamma/m*
_output_shapes
:@*
dtype0
�
Adam/b_norm_4_5/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*'
shared_nameAdam/b_norm_4_5/beta/m
}
*Adam/b_norm_4_5/beta/m/Read/ReadVariableOpReadVariableOpAdam/b_norm_4_5/beta/m*
_output_shapes
:@*
dtype0
�
*Adam/output_class_distribution_21/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
��*;
shared_name,*Adam/output_class_distribution_21/kernel/m
�
>Adam/output_class_distribution_21/kernel/m/Read/ReadVariableOpReadVariableOp*Adam/output_class_distribution_21/kernel/m* 
_output_shapes
:
��*
dtype0
�
(Adam/output_class_distribution_21/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*9
shared_name*(Adam/output_class_distribution_21/bias/m
�
<Adam/output_class_distribution_21/bias/m/Read/ReadVariableOpReadVariableOp(Adam/output_class_distribution_21/bias/m*
_output_shapes	
:�*
dtype0
�
Adam/input_conv_21/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*,
shared_nameAdam/input_conv_21/kernel/v
�
/Adam/input_conv_21/kernel/v/Read/ReadVariableOpReadVariableOpAdam/input_conv_21/kernel/v*&
_output_shapes
:*
dtype0
�
Adam/input_conv_21/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:**
shared_nameAdam/input_conv_21/bias/v
�
-Adam/input_conv_21/bias/v/Read/ReadVariableOpReadVariableOpAdam/input_conv_21/bias/v*
_output_shapes
:*
dtype0
�
)Adam/input_batch_normalization_21/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*:
shared_name+)Adam/input_batch_normalization_21/gamma/v
�
=Adam/input_batch_normalization_21/gamma/v/Read/ReadVariableOpReadVariableOp)Adam/input_batch_normalization_21/gamma/v*
_output_shapes
:*
dtype0
�
(Adam/input_batch_normalization_21/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*9
shared_name*(Adam/input_batch_normalization_21/beta/v
�
<Adam/input_batch_normalization_21/beta/v/Read/ReadVariableOpReadVariableOp(Adam/input_batch_normalization_21/beta/v*
_output_shapes
:*
dtype0
�
Adam/conv_1_21/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*(
shared_nameAdam/conv_1_21/kernel/v
�
+Adam/conv_1_21/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv_1_21/kernel/v*&
_output_shapes
:*
dtype0
�
Adam/conv_1_21/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*&
shared_nameAdam/conv_1_21/bias/v
{
)Adam/conv_1_21/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv_1_21/bias/v*
_output_shapes
:*
dtype0
�
Adam/b_norm_1_21/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*)
shared_nameAdam/b_norm_1_21/gamma/v
�
,Adam/b_norm_1_21/gamma/v/Read/ReadVariableOpReadVariableOpAdam/b_norm_1_21/gamma/v*
_output_shapes
:*
dtype0
�
Adam/b_norm_1_21/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*(
shared_nameAdam/b_norm_1_21/beta/v

+Adam/b_norm_1_21/beta/v/Read/ReadVariableOpReadVariableOpAdam/b_norm_1_21/beta/v*
_output_shapes
:*
dtype0
�
Adam/conv_2_21/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *(
shared_nameAdam/conv_2_21/kernel/v
�
+Adam/conv_2_21/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv_2_21/kernel/v*&
_output_shapes
: *
dtype0
�
Adam/conv_2_21/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *&
shared_nameAdam/conv_2_21/bias/v
{
)Adam/conv_2_21/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv_2_21/bias/v*
_output_shapes
: *
dtype0
�
Adam/b_norm_2_21/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *)
shared_nameAdam/b_norm_2_21/gamma/v
�
,Adam/b_norm_2_21/gamma/v/Read/ReadVariableOpReadVariableOpAdam/b_norm_2_21/gamma/v*
_output_shapes
: *
dtype0
�
Adam/b_norm_2_21/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *(
shared_nameAdam/b_norm_2_21/beta/v

+Adam/b_norm_2_21/beta/v/Read/ReadVariableOpReadVariableOpAdam/b_norm_2_21/beta/v*
_output_shapes
: *
dtype0
�
Adam/conv_3_13/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: @*(
shared_nameAdam/conv_3_13/kernel/v
�
+Adam/conv_3_13/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv_3_13/kernel/v*&
_output_shapes
: @*
dtype0
�
Adam/conv_3_13/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*&
shared_nameAdam/conv_3_13/bias/v
{
)Adam/conv_3_13/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv_3_13/bias/v*
_output_shapes
:@*
dtype0
�
Adam/b_norm_3_13/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*)
shared_nameAdam/b_norm_3_13/gamma/v
�
,Adam/b_norm_3_13/gamma/v/Read/ReadVariableOpReadVariableOpAdam/b_norm_3_13/gamma/v*
_output_shapes
:@*
dtype0
�
Adam/b_norm_3_13/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*(
shared_nameAdam/b_norm_3_13/beta/v

+Adam/b_norm_3_13/beta/v/Read/ReadVariableOpReadVariableOpAdam/b_norm_3_13/beta/v*
_output_shapes
:@*
dtype0
�
Adam/conv_4_5/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@@*'
shared_nameAdam/conv_4_5/kernel/v
�
*Adam/conv_4_5/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv_4_5/kernel/v*&
_output_shapes
:@@*
dtype0
�
Adam/conv_4_5/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*%
shared_nameAdam/conv_4_5/bias/v
y
(Adam/conv_4_5/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv_4_5/bias/v*
_output_shapes
:@*
dtype0
�
Adam/b_norm_4_5/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*(
shared_nameAdam/b_norm_4_5/gamma/v

+Adam/b_norm_4_5/gamma/v/Read/ReadVariableOpReadVariableOpAdam/b_norm_4_5/gamma/v*
_output_shapes
:@*
dtype0
�
Adam/b_norm_4_5/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*'
shared_nameAdam/b_norm_4_5/beta/v
}
*Adam/b_norm_4_5/beta/v/Read/ReadVariableOpReadVariableOpAdam/b_norm_4_5/beta/v*
_output_shapes
:@*
dtype0
�
*Adam/output_class_distribution_21/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
��*;
shared_name,*Adam/output_class_distribution_21/kernel/v
�
>Adam/output_class_distribution_21/kernel/v/Read/ReadVariableOpReadVariableOp*Adam/output_class_distribution_21/kernel/v* 
_output_shapes
:
��*
dtype0
�
(Adam/output_class_distribution_21/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*9
shared_name*(Adam/output_class_distribution_21/bias/v
�
<Adam/output_class_distribution_21/bias/v/Read/ReadVariableOpReadVariableOp(Adam/output_class_distribution_21/bias/v*
_output_shapes	
:�*
dtype0

NoOpNoOp
��
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*��
value�B� B�
�
layer-0
layer_with_weights-0
layer-1
layer_with_weights-1
layer-2
layer-3
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
layer_with_weights-5
layer-11
layer-12
layer-13
layer_with_weights-6
layer-14
layer_with_weights-7
layer-15
layer-16
layer-17
layer_with_weights-8
layer-18
layer_with_weights-9
layer-19
layer-20
layer-21
layer-22
layer-23
layer_with_weights-10
layer-24
	optimizer
trainable_variables
regularization_losses
	variables
	keras_api

signatures
 
h

 kernel
!bias
"trainable_variables
#regularization_losses
$	variables
%	keras_api
�
&axis
	'gamma
(beta
)moving_mean
*moving_variance
+trainable_variables
,regularization_losses
-	variables
.	keras_api
R
/trainable_variables
0regularization_losses
1	variables
2	keras_api
R
3trainable_variables
4regularization_losses
5	variables
6	keras_api
R
7trainable_variables
8regularization_losses
9	variables
:	keras_api
h

;kernel
<bias
=trainable_variables
>regularization_losses
?	variables
@	keras_api
�
Aaxis
	Bgamma
Cbeta
Dmoving_mean
Emoving_variance
Ftrainable_variables
Gregularization_losses
H	variables
I	keras_api
R
Jtrainable_variables
Kregularization_losses
L	variables
M	keras_api
R
Ntrainable_variables
Oregularization_losses
P	variables
Q	keras_api
h

Rkernel
Sbias
Ttrainable_variables
Uregularization_losses
V	variables
W	keras_api
�
Xaxis
	Ygamma
Zbeta
[moving_mean
\moving_variance
]trainable_variables
^regularization_losses
_	variables
`	keras_api
R
atrainable_variables
bregularization_losses
c	variables
d	keras_api
R
etrainable_variables
fregularization_losses
g	variables
h	keras_api
h

ikernel
jbias
ktrainable_variables
lregularization_losses
m	variables
n	keras_api
�
oaxis
	pgamma
qbeta
rmoving_mean
smoving_variance
ttrainable_variables
uregularization_losses
v	variables
w	keras_api
R
xtrainable_variables
yregularization_losses
z	variables
{	keras_api
R
|trainable_variables
}regularization_losses
~	variables
	keras_api
n
�kernel
	�bias
�trainable_variables
�regularization_losses
�	variables
�	keras_api
�
	�axis

�gamma
	�beta
�moving_mean
�moving_variance
�trainable_variables
�regularization_losses
�	variables
�	keras_api
V
�trainable_variables
�regularization_losses
�	variables
�	keras_api
V
�trainable_variables
�regularization_losses
�	variables
�	keras_api
V
�trainable_variables
�regularization_losses
�	variables
�	keras_api
V
�trainable_variables
�regularization_losses
�	variables
�	keras_api
n
�kernel
	�bias
�trainable_variables
�regularization_losses
�	variables
�	keras_api
�
	�iter
�beta_1
�beta_2

�decay
�learning_rate m�!m�'m�(m�;m�<m�Bm�Cm�Rm�Sm�Ym�Zm�im�jm�pm�qm�	�m�	�m�	�m�	�m�	�m�	�m� v�!v�'v�(v�;v�<v�Bv�Cv�Rv�Sv�Yv�Zv�iv�jv�pv�qv�	�v�	�v�	�v�	�v�	�v�	�v�
�
 0
!1
'2
(3
;4
<5
B6
C7
R8
S9
Y10
Z11
i12
j13
p14
q15
�16
�17
�18
�19
�20
�21
 
�
 0
!1
'2
(3
)4
*5
;6
<7
B8
C9
D10
E11
R12
S13
Y14
Z15
[16
\17
i18
j19
p20
q21
r22
s23
�24
�25
�26
�27
�28
�29
�30
�31
�
 �layer_regularization_losses
trainable_variables
regularization_losses
�non_trainable_variables
�layers
	variables
�metrics
 
`^
VARIABLE_VALUEinput_conv_21/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE
\Z
VARIABLE_VALUEinput_conv_21/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE

 0
!1
 

 0
!1
�
�metrics
"trainable_variables
#regularization_losses
�non_trainable_variables
�layers
$	variables
 �layer_regularization_losses
 
mk
VARIABLE_VALUE"input_batch_normalization_21/gamma5layer_with_weights-1/gamma/.ATTRIBUTES/VARIABLE_VALUE
ki
VARIABLE_VALUE!input_batch_normalization_21/beta4layer_with_weights-1/beta/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUE(input_batch_normalization_21/moving_mean;layer_with_weights-1/moving_mean/.ATTRIBUTES/VARIABLE_VALUE
�
VARIABLE_VALUE,input_batch_normalization_21/moving_variance?layer_with_weights-1/moving_variance/.ATTRIBUTES/VARIABLE_VALUE

'0
(1
 

'0
(1
)2
*3
�
�metrics
+trainable_variables
,regularization_losses
�non_trainable_variables
�layers
-	variables
 �layer_regularization_losses
 
 
 
�
�metrics
/trainable_variables
0regularization_losses
�non_trainable_variables
�layers
1	variables
 �layer_regularization_losses
 
 
 
�
�metrics
3trainable_variables
4regularization_losses
�non_trainable_variables
�layers
5	variables
 �layer_regularization_losses
 
 
 
�
�metrics
7trainable_variables
8regularization_losses
�non_trainable_variables
�layers
9	variables
 �layer_regularization_losses
\Z
VARIABLE_VALUEconv_1_21/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEconv_1_21/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE

;0
<1
 

;0
<1
�
�metrics
=trainable_variables
>regularization_losses
�non_trainable_variables
�layers
?	variables
 �layer_regularization_losses
 
\Z
VARIABLE_VALUEb_norm_1_21/gamma5layer_with_weights-3/gamma/.ATTRIBUTES/VARIABLE_VALUE
ZX
VARIABLE_VALUEb_norm_1_21/beta4layer_with_weights-3/beta/.ATTRIBUTES/VARIABLE_VALUE
hf
VARIABLE_VALUEb_norm_1_21/moving_mean;layer_with_weights-3/moving_mean/.ATTRIBUTES/VARIABLE_VALUE
pn
VARIABLE_VALUEb_norm_1_21/moving_variance?layer_with_weights-3/moving_variance/.ATTRIBUTES/VARIABLE_VALUE

B0
C1
 

B0
C1
D2
E3
�
�metrics
Ftrainable_variables
Gregularization_losses
�non_trainable_variables
�layers
H	variables
 �layer_regularization_losses
 
 
 
�
�metrics
Jtrainable_variables
Kregularization_losses
�non_trainable_variables
�layers
L	variables
 �layer_regularization_losses
 
 
 
�
�metrics
Ntrainable_variables
Oregularization_losses
�non_trainable_variables
�layers
P	variables
 �layer_regularization_losses
\Z
VARIABLE_VALUEconv_2_21/kernel6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEconv_2_21/bias4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUE

R0
S1
 

R0
S1
�
�metrics
Ttrainable_variables
Uregularization_losses
�non_trainable_variables
�layers
V	variables
 �layer_regularization_losses
 
\Z
VARIABLE_VALUEb_norm_2_21/gamma5layer_with_weights-5/gamma/.ATTRIBUTES/VARIABLE_VALUE
ZX
VARIABLE_VALUEb_norm_2_21/beta4layer_with_weights-5/beta/.ATTRIBUTES/VARIABLE_VALUE
hf
VARIABLE_VALUEb_norm_2_21/moving_mean;layer_with_weights-5/moving_mean/.ATTRIBUTES/VARIABLE_VALUE
pn
VARIABLE_VALUEb_norm_2_21/moving_variance?layer_with_weights-5/moving_variance/.ATTRIBUTES/VARIABLE_VALUE

Y0
Z1
 

Y0
Z1
[2
\3
�
�metrics
]trainable_variables
^regularization_losses
�non_trainable_variables
�layers
_	variables
 �layer_regularization_losses
 
 
 
�
�metrics
atrainable_variables
bregularization_losses
�non_trainable_variables
�layers
c	variables
 �layer_regularization_losses
 
 
 
�
�metrics
etrainable_variables
fregularization_losses
�non_trainable_variables
�layers
g	variables
 �layer_regularization_losses
\Z
VARIABLE_VALUEconv_3_13/kernel6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEconv_3_13/bias4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUE

i0
j1
 

i0
j1
�
�metrics
ktrainable_variables
lregularization_losses
�non_trainable_variables
�layers
m	variables
 �layer_regularization_losses
 
\Z
VARIABLE_VALUEb_norm_3_13/gamma5layer_with_weights-7/gamma/.ATTRIBUTES/VARIABLE_VALUE
ZX
VARIABLE_VALUEb_norm_3_13/beta4layer_with_weights-7/beta/.ATTRIBUTES/VARIABLE_VALUE
hf
VARIABLE_VALUEb_norm_3_13/moving_mean;layer_with_weights-7/moving_mean/.ATTRIBUTES/VARIABLE_VALUE
pn
VARIABLE_VALUEb_norm_3_13/moving_variance?layer_with_weights-7/moving_variance/.ATTRIBUTES/VARIABLE_VALUE

p0
q1
 

p0
q1
r2
s3
�
�metrics
ttrainable_variables
uregularization_losses
�non_trainable_variables
�layers
v	variables
 �layer_regularization_losses
 
 
 
�
�metrics
xtrainable_variables
yregularization_losses
�non_trainable_variables
�layers
z	variables
 �layer_regularization_losses
 
 
 
�
�metrics
|trainable_variables
}regularization_losses
�non_trainable_variables
�layers
~	variables
 �layer_regularization_losses
[Y
VARIABLE_VALUEconv_4_5/kernel6layer_with_weights-8/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEconv_4_5/bias4layer_with_weights-8/bias/.ATTRIBUTES/VARIABLE_VALUE

�0
�1
 

�0
�1
�
�metrics
�trainable_variables
�regularization_losses
�non_trainable_variables
�layers
�	variables
 �layer_regularization_losses
 
[Y
VARIABLE_VALUEb_norm_4_5/gamma5layer_with_weights-9/gamma/.ATTRIBUTES/VARIABLE_VALUE
YW
VARIABLE_VALUEb_norm_4_5/beta4layer_with_weights-9/beta/.ATTRIBUTES/VARIABLE_VALUE
ge
VARIABLE_VALUEb_norm_4_5/moving_mean;layer_with_weights-9/moving_mean/.ATTRIBUTES/VARIABLE_VALUE
om
VARIABLE_VALUEb_norm_4_5/moving_variance?layer_with_weights-9/moving_variance/.ATTRIBUTES/VARIABLE_VALUE

�0
�1
 
 
�0
�1
�2
�3
�
�metrics
�trainable_variables
�regularization_losses
�non_trainable_variables
�layers
�	variables
 �layer_regularization_losses
 
 
 
�
�metrics
�trainable_variables
�regularization_losses
�non_trainable_variables
�layers
�	variables
 �layer_regularization_losses
 
 
 
�
�metrics
�trainable_variables
�regularization_losses
�non_trainable_variables
�layers
�	variables
 �layer_regularization_losses
 
 
 
�
�metrics
�trainable_variables
�regularization_losses
�non_trainable_variables
�layers
�	variables
 �layer_regularization_losses
 
 
 
�
�metrics
�trainable_variables
�regularization_losses
�non_trainable_variables
�layers
�	variables
 �layer_regularization_losses
pn
VARIABLE_VALUE#output_class_distribution_21/kernel7layer_with_weights-10/kernel/.ATTRIBUTES/VARIABLE_VALUE
lj
VARIABLE_VALUE!output_class_distribution_21/bias5layer_with_weights-10/bias/.ATTRIBUTES/VARIABLE_VALUE

�0
�1
 

�0
�1
�
�metrics
�trainable_variables
�regularization_losses
�non_trainable_variables
�layers
�	variables
 �layer_regularization_losses
HF
VARIABLE_VALUE	Adam/iter)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEAdam/beta_1+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEAdam/beta_2+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUE
JH
VARIABLE_VALUE
Adam/decay*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUE
ZX
VARIABLE_VALUEAdam/learning_rate2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUE
 
H
)0
*1
D2
E3
[4
\5
r6
s7
�8
�9
�
0
1
2
3
4
5
6
	7

8
9
10
11
12
13
14
15
16
17
18
19
20
21
22
23

�0
 
 
 
 
 

)0
*1
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 

D0
E1
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 

[0
\1
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 

r0
s1
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 

�0
�1
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 


�total

�count
�
_fn_kwargs
�trainable_variables
�regularization_losses
�	variables
�	keras_api
OM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE
OM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE
 
 
 

�0
�1
�
�metrics
�trainable_variables
�regularization_losses
�non_trainable_variables
�layers
�	variables
 �layer_regularization_losses
 

�0
�1
 
 
��
VARIABLE_VALUEAdam/input_conv_21/kernel/mRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/input_conv_21/bias/mPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUE)Adam/input_batch_normalization_21/gamma/mQlayer_with_weights-1/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUE(Adam/input_batch_normalization_21/beta/mPlayer_with_weights-1/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/conv_1_21/kernel/mRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/conv_1_21/bias/mPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/b_norm_1_21/gamma/mQlayer_with_weights-3/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}{
VARIABLE_VALUEAdam/b_norm_1_21/beta/mPlayer_with_weights-3/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/conv_2_21/kernel/mRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/conv_2_21/bias/mPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/b_norm_2_21/gamma/mQlayer_with_weights-5/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}{
VARIABLE_VALUEAdam/b_norm_2_21/beta/mPlayer_with_weights-5/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/conv_3_13/kernel/mRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/conv_3_13/bias/mPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/b_norm_3_13/gamma/mQlayer_with_weights-7/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}{
VARIABLE_VALUEAdam/b_norm_3_13/beta/mPlayer_with_weights-7/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/conv_4_5/kernel/mRlayer_with_weights-8/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/conv_4_5/bias/mPlayer_with_weights-8/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/b_norm_4_5/gamma/mQlayer_with_weights-9/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUEAdam/b_norm_4_5/beta/mPlayer_with_weights-9/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUE*Adam/output_class_distribution_21/kernel/mSlayer_with_weights-10/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUE(Adam/output_class_distribution_21/bias/mQlayer_with_weights-10/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUEAdam/input_conv_21/kernel/vRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/input_conv_21/bias/vPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUE)Adam/input_batch_normalization_21/gamma/vQlayer_with_weights-1/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUE(Adam/input_batch_normalization_21/beta/vPlayer_with_weights-1/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/conv_1_21/kernel/vRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/conv_1_21/bias/vPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/b_norm_1_21/gamma/vQlayer_with_weights-3/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}{
VARIABLE_VALUEAdam/b_norm_1_21/beta/vPlayer_with_weights-3/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/conv_2_21/kernel/vRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/conv_2_21/bias/vPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/b_norm_2_21/gamma/vQlayer_with_weights-5/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}{
VARIABLE_VALUEAdam/b_norm_2_21/beta/vPlayer_with_weights-5/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/conv_3_13/kernel/vRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/conv_3_13/bias/vPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/b_norm_3_13/gamma/vQlayer_with_weights-7/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}{
VARIABLE_VALUEAdam/b_norm_3_13/beta/vPlayer_with_weights-7/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/conv_4_5/kernel/vRlayer_with_weights-8/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/conv_4_5/bias/vPlayer_with_weights-8/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/b_norm_4_5/gamma/vQlayer_with_weights-9/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUEAdam/b_norm_4_5/beta/vPlayer_with_weights-9/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUE*Adam/output_class_distribution_21/kernel/vSlayer_with_weights-10/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUE(Adam/output_class_distribution_21/bias/vQlayer_with_weights-10/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
�
 serving_default_input_conv_inputPlaceholder*1
_output_shapes
:�����������*
dtype0*&
shape:�����������
�
StatefulPartitionedCallStatefulPartitionedCall serving_default_input_conv_inputinput_conv_21/kernelinput_conv_21/bias"input_batch_normalization_21/gamma!input_batch_normalization_21/beta(input_batch_normalization_21/moving_mean,input_batch_normalization_21/moving_varianceconv_1_21/kernelconv_1_21/biasb_norm_1_21/gammab_norm_1_21/betab_norm_1_21/moving_meanb_norm_1_21/moving_varianceconv_2_21/kernelconv_2_21/biasb_norm_2_21/gammab_norm_2_21/betab_norm_2_21/moving_meanb_norm_2_21/moving_varianceconv_3_13/kernelconv_3_13/biasb_norm_3_13/gammab_norm_3_13/betab_norm_3_13/moving_meanb_norm_3_13/moving_varianceconv_4_5/kernelconv_4_5/biasb_norm_4_5/gammab_norm_4_5/betab_norm_4_5/moving_meanb_norm_4_5/moving_variance#output_class_distribution_21/kernel!output_class_distribution_21/bias*,
Tin%
#2!*
Tout
2*,
_gradient_op_typePartitionedCallUnused*(
_output_shapes
:����������*-
config_proto

GPU

CPU2*0J 8*.
f)R'
%__inference_signature_wrapper_4353810
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
� 
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename(input_conv_21/kernel/Read/ReadVariableOp&input_conv_21/bias/Read/ReadVariableOp6input_batch_normalization_21/gamma/Read/ReadVariableOp5input_batch_normalization_21/beta/Read/ReadVariableOp<input_batch_normalization_21/moving_mean/Read/ReadVariableOp@input_batch_normalization_21/moving_variance/Read/ReadVariableOp$conv_1_21/kernel/Read/ReadVariableOp"conv_1_21/bias/Read/ReadVariableOp%b_norm_1_21/gamma/Read/ReadVariableOp$b_norm_1_21/beta/Read/ReadVariableOp+b_norm_1_21/moving_mean/Read/ReadVariableOp/b_norm_1_21/moving_variance/Read/ReadVariableOp$conv_2_21/kernel/Read/ReadVariableOp"conv_2_21/bias/Read/ReadVariableOp%b_norm_2_21/gamma/Read/ReadVariableOp$b_norm_2_21/beta/Read/ReadVariableOp+b_norm_2_21/moving_mean/Read/ReadVariableOp/b_norm_2_21/moving_variance/Read/ReadVariableOp$conv_3_13/kernel/Read/ReadVariableOp"conv_3_13/bias/Read/ReadVariableOp%b_norm_3_13/gamma/Read/ReadVariableOp$b_norm_3_13/beta/Read/ReadVariableOp+b_norm_3_13/moving_mean/Read/ReadVariableOp/b_norm_3_13/moving_variance/Read/ReadVariableOp#conv_4_5/kernel/Read/ReadVariableOp!conv_4_5/bias/Read/ReadVariableOp$b_norm_4_5/gamma/Read/ReadVariableOp#b_norm_4_5/beta/Read/ReadVariableOp*b_norm_4_5/moving_mean/Read/ReadVariableOp.b_norm_4_5/moving_variance/Read/ReadVariableOp7output_class_distribution_21/kernel/Read/ReadVariableOp5output_class_distribution_21/bias/Read/ReadVariableOpAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOp/Adam/input_conv_21/kernel/m/Read/ReadVariableOp-Adam/input_conv_21/bias/m/Read/ReadVariableOp=Adam/input_batch_normalization_21/gamma/m/Read/ReadVariableOp<Adam/input_batch_normalization_21/beta/m/Read/ReadVariableOp+Adam/conv_1_21/kernel/m/Read/ReadVariableOp)Adam/conv_1_21/bias/m/Read/ReadVariableOp,Adam/b_norm_1_21/gamma/m/Read/ReadVariableOp+Adam/b_norm_1_21/beta/m/Read/ReadVariableOp+Adam/conv_2_21/kernel/m/Read/ReadVariableOp)Adam/conv_2_21/bias/m/Read/ReadVariableOp,Adam/b_norm_2_21/gamma/m/Read/ReadVariableOp+Adam/b_norm_2_21/beta/m/Read/ReadVariableOp+Adam/conv_3_13/kernel/m/Read/ReadVariableOp)Adam/conv_3_13/bias/m/Read/ReadVariableOp,Adam/b_norm_3_13/gamma/m/Read/ReadVariableOp+Adam/b_norm_3_13/beta/m/Read/ReadVariableOp*Adam/conv_4_5/kernel/m/Read/ReadVariableOp(Adam/conv_4_5/bias/m/Read/ReadVariableOp+Adam/b_norm_4_5/gamma/m/Read/ReadVariableOp*Adam/b_norm_4_5/beta/m/Read/ReadVariableOp>Adam/output_class_distribution_21/kernel/m/Read/ReadVariableOp<Adam/output_class_distribution_21/bias/m/Read/ReadVariableOp/Adam/input_conv_21/kernel/v/Read/ReadVariableOp-Adam/input_conv_21/bias/v/Read/ReadVariableOp=Adam/input_batch_normalization_21/gamma/v/Read/ReadVariableOp<Adam/input_batch_normalization_21/beta/v/Read/ReadVariableOp+Adam/conv_1_21/kernel/v/Read/ReadVariableOp)Adam/conv_1_21/bias/v/Read/ReadVariableOp,Adam/b_norm_1_21/gamma/v/Read/ReadVariableOp+Adam/b_norm_1_21/beta/v/Read/ReadVariableOp+Adam/conv_2_21/kernel/v/Read/ReadVariableOp)Adam/conv_2_21/bias/v/Read/ReadVariableOp,Adam/b_norm_2_21/gamma/v/Read/ReadVariableOp+Adam/b_norm_2_21/beta/v/Read/ReadVariableOp+Adam/conv_3_13/kernel/v/Read/ReadVariableOp)Adam/conv_3_13/bias/v/Read/ReadVariableOp,Adam/b_norm_3_13/gamma/v/Read/ReadVariableOp+Adam/b_norm_3_13/beta/v/Read/ReadVariableOp*Adam/conv_4_5/kernel/v/Read/ReadVariableOp(Adam/conv_4_5/bias/v/Read/ReadVariableOp+Adam/b_norm_4_5/gamma/v/Read/ReadVariableOp*Adam/b_norm_4_5/beta/v/Read/ReadVariableOp>Adam/output_class_distribution_21/kernel/v/Read/ReadVariableOp<Adam/output_class_distribution_21/bias/v/Read/ReadVariableOpConst*`
TinY
W2U	*
Tout
2*,
_gradient_op_typePartitionedCallUnused*
_output_shapes
: *-
config_proto

GPU

CPU2*0J 8*)
f$R"
 __inference__traced_save_4355483
�
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenameinput_conv_21/kernelinput_conv_21/bias"input_batch_normalization_21/gamma!input_batch_normalization_21/beta(input_batch_normalization_21/moving_mean,input_batch_normalization_21/moving_varianceconv_1_21/kernelconv_1_21/biasb_norm_1_21/gammab_norm_1_21/betab_norm_1_21/moving_meanb_norm_1_21/moving_varianceconv_2_21/kernelconv_2_21/biasb_norm_2_21/gammab_norm_2_21/betab_norm_2_21/moving_meanb_norm_2_21/moving_varianceconv_3_13/kernelconv_3_13/biasb_norm_3_13/gammab_norm_3_13/betab_norm_3_13/moving_meanb_norm_3_13/moving_varianceconv_4_5/kernelconv_4_5/biasb_norm_4_5/gammab_norm_4_5/betab_norm_4_5/moving_meanb_norm_4_5/moving_variance#output_class_distribution_21/kernel!output_class_distribution_21/bias	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_ratetotalcountAdam/input_conv_21/kernel/mAdam/input_conv_21/bias/m)Adam/input_batch_normalization_21/gamma/m(Adam/input_batch_normalization_21/beta/mAdam/conv_1_21/kernel/mAdam/conv_1_21/bias/mAdam/b_norm_1_21/gamma/mAdam/b_norm_1_21/beta/mAdam/conv_2_21/kernel/mAdam/conv_2_21/bias/mAdam/b_norm_2_21/gamma/mAdam/b_norm_2_21/beta/mAdam/conv_3_13/kernel/mAdam/conv_3_13/bias/mAdam/b_norm_3_13/gamma/mAdam/b_norm_3_13/beta/mAdam/conv_4_5/kernel/mAdam/conv_4_5/bias/mAdam/b_norm_4_5/gamma/mAdam/b_norm_4_5/beta/m*Adam/output_class_distribution_21/kernel/m(Adam/output_class_distribution_21/bias/mAdam/input_conv_21/kernel/vAdam/input_conv_21/bias/v)Adam/input_batch_normalization_21/gamma/v(Adam/input_batch_normalization_21/beta/vAdam/conv_1_21/kernel/vAdam/conv_1_21/bias/vAdam/b_norm_1_21/gamma/vAdam/b_norm_1_21/beta/vAdam/conv_2_21/kernel/vAdam/conv_2_21/bias/vAdam/b_norm_2_21/gamma/vAdam/b_norm_2_21/beta/vAdam/conv_3_13/kernel/vAdam/conv_3_13/bias/vAdam/b_norm_3_13/gamma/vAdam/b_norm_3_13/beta/vAdam/conv_4_5/kernel/vAdam/conv_4_5/bias/vAdam/b_norm_4_5/gamma/vAdam/b_norm_4_5/beta/v*Adam/output_class_distribution_21/kernel/v(Adam/output_class_distribution_21/bias/v*_
TinX
V2T*
Tout
2*,
_gradient_op_typePartitionedCallUnused*
_output_shapes
: *-
config_proto

GPU

CPU2*0J 8*,
f'R%
#__inference__traced_restore_4355744ƶ
�
c
G__inference_flatten_21_layer_call_and_return_conditional_losses_4353380

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"����   2
Consth
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:����������2	
Reshapee
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:����������2

Identity"
identityIdentity:output:0*.
_input_shapes
:���������@:& "
 
_user_specified_nameinputs
�
N
2__inference_max_pooling2d_85_layer_call_fn_4352196

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*J
_output_shapes8
6:4������������������������������������*-
config_proto

GPU

CPU2*0J 8*V
fQRO
M__inference_max_pooling2d_85_layer_call_and_return_conditional_losses_43521902
PartitionedCall�
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4������������������������������������2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4������������������������������������:& "
 
_user_specified_nameinputs
�
f
G__inference_dropout_42_layer_call_and_return_conditional_losses_4354452

inputs
identity�a
dropout/rateConst*
_output_shapes
: *
dtype0*
valueB
 *33�>2
dropout/rateT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape}
dropout/random_uniform/minConst*
_output_shapes
: *
dtype0*
valueB
 *    2
dropout/random_uniform/min}
dropout/random_uniform/maxConst*
_output_shapes
: *
dtype0*
valueB
 *  �?2
dropout/random_uniform/max�
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*/
_output_shapes
:���������@@*
dtype02&
$dropout/random_uniform/RandomUniform�
dropout/random_uniform/subSub#dropout/random_uniform/max:output:0#dropout/random_uniform/min:output:0*
T0*
_output_shapes
: 2
dropout/random_uniform/sub�
dropout/random_uniform/mulMul-dropout/random_uniform/RandomUniform:output:0dropout/random_uniform/sub:z:0*
T0*/
_output_shapes
:���������@@2
dropout/random_uniform/mul�
dropout/random_uniformAdddropout/random_uniform/mul:z:0#dropout/random_uniform/min:output:0*
T0*/
_output_shapes
:���������@@2
dropout/random_uniformc
dropout/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?2
dropout/sub/xq
dropout/subSubdropout/sub/x:output:0dropout/rate:output:0*
T0*
_output_shapes
: 2
dropout/subk
dropout/truediv/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?2
dropout/truediv/x{
dropout/truedivRealDivdropout/truediv/x:output:0dropout/sub:z:0*
T0*
_output_shapes
: 2
dropout/truediv�
dropout/GreaterEqualGreaterEqualdropout/random_uniform:z:0dropout/rate:output:0*
T0*/
_output_shapes
:���������@@2
dropout/GreaterEqualx
dropout/mulMulinputsdropout/truediv:z:0*
T0*/
_output_shapes
:���������@@2
dropout/mul�
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:���������@@2
dropout/Cast�
dropout/mul_1Muldropout/mul:z:0dropout/Cast:y:0*
T0*/
_output_shapes
:���������@@2
dropout/mul_1m
IdentityIdentitydropout/mul_1:z:0*
T0*/
_output_shapes
:���������@@2

Identity"
identityIdentity:output:0*.
_input_shapes
:���������@@:& "
 
_user_specified_nameinputs
�
e
G__inference_dropout_42_layer_call_and_return_conditional_losses_4352977

inputs

identity_1b
IdentityIdentityinputs*
T0*/
_output_shapes
:���������@@2

Identityq

Identity_1IdentityIdentity:output:0*
T0*/
_output_shapes
:���������@@2

Identity_1"!

identity_1Identity_1:output:0*.
_input_shapes
:���������@@:& "
 
_user_specified_nameinputs
�

�
G__inference_input_conv_layer_call_and_return_conditional_losses_4352044

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOpo
dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      2
dilation_rate�
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype02
Conv2D/ReadVariableOp�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+���������������������������*
paddingSAME*
strides
2
Conv2D�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+���������������������������2	
BiasAdd�
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*A
_output_shapes/
-:+���������������������������2

Identity"
identityIdentity:output:0*H
_input_shapes7
5:+���������������������������::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:& "
 
_user_specified_nameinputs
�
i
M__inference_max_pooling2d_81_layer_call_and_return_conditional_losses_4352354

inputs
identity�
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4������������������������������������*
ksize
*
paddingVALID*
strides
2	
MaxPool�
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4������������������������������������2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4������������������������������������:& "
 
_user_specified_nameinputs
�
�	
,__inference_awsome_net_layer_call_fn_4354225

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4"
statefulpartitionedcall_args_5"
statefulpartitionedcall_args_6"
statefulpartitionedcall_args_7"
statefulpartitionedcall_args_8"
statefulpartitionedcall_args_9#
statefulpartitionedcall_args_10#
statefulpartitionedcall_args_11#
statefulpartitionedcall_args_12#
statefulpartitionedcall_args_13#
statefulpartitionedcall_args_14#
statefulpartitionedcall_args_15#
statefulpartitionedcall_args_16#
statefulpartitionedcall_args_17#
statefulpartitionedcall_args_18#
statefulpartitionedcall_args_19#
statefulpartitionedcall_args_20#
statefulpartitionedcall_args_21#
statefulpartitionedcall_args_22#
statefulpartitionedcall_args_23#
statefulpartitionedcall_args_24#
statefulpartitionedcall_args_25#
statefulpartitionedcall_args_26#
statefulpartitionedcall_args_27#
statefulpartitionedcall_args_28#
statefulpartitionedcall_args_29#
statefulpartitionedcall_args_30#
statefulpartitionedcall_args_31#
statefulpartitionedcall_args_32
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4statefulpartitionedcall_args_5statefulpartitionedcall_args_6statefulpartitionedcall_args_7statefulpartitionedcall_args_8statefulpartitionedcall_args_9statefulpartitionedcall_args_10statefulpartitionedcall_args_11statefulpartitionedcall_args_12statefulpartitionedcall_args_13statefulpartitionedcall_args_14statefulpartitionedcall_args_15statefulpartitionedcall_args_16statefulpartitionedcall_args_17statefulpartitionedcall_args_18statefulpartitionedcall_args_19statefulpartitionedcall_args_20statefulpartitionedcall_args_21statefulpartitionedcall_args_22statefulpartitionedcall_args_23statefulpartitionedcall_args_24statefulpartitionedcall_args_25statefulpartitionedcall_args_26statefulpartitionedcall_args_27statefulpartitionedcall_args_28statefulpartitionedcall_args_29statefulpartitionedcall_args_30statefulpartitionedcall_args_31statefulpartitionedcall_args_32*,
Tin%
#2!*
Tout
2*,
_gradient_op_typePartitionedCallUnused*(
_output_shapes
:����������*-
config_proto

GPU

CPU2*0J 8*P
fKRI
G__inference_awsome_net_layer_call_and_return_conditional_losses_43535722
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:����������2

Identity"
identityIdentity:output:0*�
_input_shapes�
�:�����������::::::::::::::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs
�
H
,__inference_input_RELU_layer_call_fn_4354432

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*1
_output_shapes
:�����������*-
config_proto

GPU

CPU2*0J 8*P
fKRI
G__inference_input_RELU_layer_call_and_return_conditional_losses_43529432
PartitionedCallv
IdentityIdentityPartitionedCall:output:0*
T0*1
_output_shapes
:�����������2

Identity"
identityIdentity:output:0*0
_input_shapes
:�����������:& "
 
_user_specified_nameinputs
�

�
C__inference_conv_4_layer_call_and_return_conditional_losses_4352700

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOpo
dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      2
dilation_rate�
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02
Conv2D/ReadVariableOp�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+���������������������������@*
paddingSAME*
strides
2
Conv2D�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+���������������������������@2	
BiasAdd�
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*A
_output_shapes/
-:+���������������������������@2

Identity"
identityIdentity:output:0*H
_input_shapes7
5:+���������������������������@::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:& "
 
_user_specified_nameinputs
�

�
C__inference_conv_2_layer_call_and_return_conditional_losses_4352372

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOpo
dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      2
dilation_rate�
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: *
dtype02
Conv2D/ReadVariableOp�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+��������������������������� *
paddingSAME*
strides
2
Conv2D�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+��������������������������� 2	
BiasAdd�
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*A
_output_shapes/
-:+��������������������������� 2

Identity"
identityIdentity:output:0*H
_input_shapes7
5:+���������������������������::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:& "
 
_user_specified_nameinputs
�
i
M__inference_max_pooling2d_85_layer_call_and_return_conditional_losses_4352190

inputs
identity�
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4������������������������������������*
ksize
*
paddingVALID*
strides
2	
MaxPool�
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4������������������������������������2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4������������������������������������:& "
 
_user_specified_nameinputs
�
�
*__inference_b_norm_4_layer_call_fn_4355054

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4*
Tin	
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*A
_output_shapes/
-:+���������������������������@*-
config_proto

GPU

CPU2*0J 8*N
fIRG
E__inference_b_norm_4_layer_call_and_return_conditional_losses_43528022
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+���������������������������@2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+���������������������������@::::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs
�
�
V__inference_input_batch_normalization_layer_call_and_return_conditional_losses_4352177

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity��FusedBatchNormV3/ReadVariableOp�!FusedBatchNormV3/ReadVariableOp_1�ReadVariableOp�ReadVariableOp_1^
LogicalAnd/xConst*
_output_shapes
: *
dtype0
*
value	B
 Z 2
LogicalAnd/x^
LogicalAnd/yConst*
_output_shapes
: *
dtype0
*
value	B
 Z2
LogicalAnd/yl

LogicalAnd
LogicalAndLogicalAnd/x:output:0LogicalAnd/y:output:0*
_output_shapes
: 2

LogicalAndt
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOp�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+���������������������������:::::*
epsilon%o�:*
is_training( 2
FusedBatchNormV3S
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *�p}?2
Const�
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*A
_output_shapes/
-:+���������������������������2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+���������������������������::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:& "
 
_user_specified_nameinputs
�
e
G__inference_dropout_43_layer_call_and_return_conditional_losses_4353413

inputs

identity_1[
IdentityIdentityinputs*
T0*(
_output_shapes
:����������2

Identityj

Identity_1IdentityIdentity:output:0*
T0*(
_output_shapes
:����������2

Identity_1"!

identity_1Identity_1:output:0*'
_input_shapes
:����������:& "
 
_user_specified_nameinputs
�$
�
V__inference_input_batch_normalization_layer_call_and_return_conditional_losses_4354382

inputs
readvariableop_resource
readvariableop_1_resource
assignmovingavg_4354367
assignmovingavg_1_4354374
identity��#AssignMovingAvg/AssignSubVariableOp�AssignMovingAvg/ReadVariableOp�%AssignMovingAvg_1/AssignSubVariableOp� AssignMovingAvg_1/ReadVariableOp�ReadVariableOp�ReadVariableOp_1^
LogicalAnd/xConst*
_output_shapes
: *
dtype0
*
value	B
 Z2
LogicalAnd/x^
LogicalAnd/yConst*
_output_shapes
: *
dtype0
*
value	B
 Z2
LogicalAnd/yl

LogicalAnd
LogicalAndLogicalAnd/x:output:0LogicalAnd/y:output:0*
_output_shapes
: 2

LogicalAndt
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1Q
ConstConst*
_output_shapes
: *
dtype0*
valueB 2
ConstU
Const_1Const*
_output_shapes
: *
dtype0*
valueB 2	
Const_1�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0Const:output:0Const_1:output:0*
T0*
U0*M
_output_shapes;
9:�����������:::::*
epsilon%o�:2
FusedBatchNormV3W
Const_2Const*
_output_shapes
: *
dtype0*
valueB
 *�p}?2	
Const_2�
AssignMovingAvg/sub/xConst**
_class 
loc:@AssignMovingAvg/4354367*
_output_shapes
: *
dtype0*
valueB
 *  �?2
AssignMovingAvg/sub/x�
AssignMovingAvg/subSubAssignMovingAvg/sub/x:output:0Const_2:output:0*
T0**
_class 
loc:@AssignMovingAvg/4354367*
_output_shapes
: 2
AssignMovingAvg/sub�
AssignMovingAvg/ReadVariableOpReadVariableOpassignmovingavg_4354367*
_output_shapes
:*
dtype02 
AssignMovingAvg/ReadVariableOp�
AssignMovingAvg/sub_1Sub&AssignMovingAvg/ReadVariableOp:value:0FusedBatchNormV3:batch_mean:0*
T0**
_class 
loc:@AssignMovingAvg/4354367*
_output_shapes
:2
AssignMovingAvg/sub_1�
AssignMovingAvg/mulMulAssignMovingAvg/sub_1:z:0AssignMovingAvg/sub:z:0*
T0**
_class 
loc:@AssignMovingAvg/4354367*
_output_shapes
:2
AssignMovingAvg/mul�
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOpassignmovingavg_4354367AssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp**
_class 
loc:@AssignMovingAvg/4354367*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOp�
AssignMovingAvg_1/sub/xConst*,
_class"
 loc:@AssignMovingAvg_1/4354374*
_output_shapes
: *
dtype0*
valueB
 *  �?2
AssignMovingAvg_1/sub/x�
AssignMovingAvg_1/subSub AssignMovingAvg_1/sub/x:output:0Const_2:output:0*
T0*,
_class"
 loc:@AssignMovingAvg_1/4354374*
_output_shapes
: 2
AssignMovingAvg_1/sub�
 AssignMovingAvg_1/ReadVariableOpReadVariableOpassignmovingavg_1_4354374*
_output_shapes
:*
dtype02"
 AssignMovingAvg_1/ReadVariableOp�
AssignMovingAvg_1/sub_1Sub(AssignMovingAvg_1/ReadVariableOp:value:0!FusedBatchNormV3:batch_variance:0*
T0*,
_class"
 loc:@AssignMovingAvg_1/4354374*
_output_shapes
:2
AssignMovingAvg_1/sub_1�
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub_1:z:0AssignMovingAvg_1/sub:z:0*
T0*,
_class"
 loc:@AssignMovingAvg_1/4354374*
_output_shapes
:2
AssignMovingAvg_1/mul�
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOpassignmovingavg_1_4354374AssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*,
_class"
 loc:@AssignMovingAvg_1/4354374*
_output_shapes
 *
dtype02'
%AssignMovingAvg_1/AssignSubVariableOp�
IdentityIdentityFusedBatchNormV3:y:0$^AssignMovingAvg/AssignSubVariableOp^AssignMovingAvg/ReadVariableOp&^AssignMovingAvg_1/AssignSubVariableOp!^AssignMovingAvg_1/ReadVariableOp^ReadVariableOp^ReadVariableOp_1*
T0*1
_output_shapes
:�����������2

Identity"
identityIdentity:output:0*@
_input_shapes/
-:�����������::::2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp2D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp2 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:& "
 
_user_specified_nameinputs
�
�
*__inference_b_norm_3_layer_call_fn_4354967

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4*
Tin	
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*/
_output_shapes
:���������@*-
config_proto

GPU

CPU2*0J 8*N
fIRG
E__inference_b_norm_3_layer_call_and_return_conditional_losses_43532402
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:���������@2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:���������@::::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs
�$
�
E__inference_b_norm_2_layer_call_and_return_conditional_losses_4354757

inputs
readvariableop_resource
readvariableop_1_resource
assignmovingavg_4354742
assignmovingavg_1_4354749
identity��#AssignMovingAvg/AssignSubVariableOp�AssignMovingAvg/ReadVariableOp�%AssignMovingAvg_1/AssignSubVariableOp� AssignMovingAvg_1/ReadVariableOp�ReadVariableOp�ReadVariableOp_1^
LogicalAnd/xConst*
_output_shapes
: *
dtype0
*
value	B
 Z2
LogicalAnd/x^
LogicalAnd/yConst*
_output_shapes
: *
dtype0
*
value	B
 Z2
LogicalAnd/yl

LogicalAnd
LogicalAndLogicalAnd/x:output:0LogicalAnd/y:output:0*
_output_shapes
: 2

LogicalAndt
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype02
ReadVariableOp_1Q
ConstConst*
_output_shapes
: *
dtype0*
valueB 2
ConstU
Const_1Const*
_output_shapes
: *
dtype0*
valueB 2	
Const_1�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0Const:output:0Const_1:output:0*
T0*
U0*K
_output_shapes9
7:���������   : : : : :*
epsilon%o�:2
FusedBatchNormV3W
Const_2Const*
_output_shapes
: *
dtype0*
valueB
 *�p}?2	
Const_2�
AssignMovingAvg/sub/xConst**
_class 
loc:@AssignMovingAvg/4354742*
_output_shapes
: *
dtype0*
valueB
 *  �?2
AssignMovingAvg/sub/x�
AssignMovingAvg/subSubAssignMovingAvg/sub/x:output:0Const_2:output:0*
T0**
_class 
loc:@AssignMovingAvg/4354742*
_output_shapes
: 2
AssignMovingAvg/sub�
AssignMovingAvg/ReadVariableOpReadVariableOpassignmovingavg_4354742*
_output_shapes
: *
dtype02 
AssignMovingAvg/ReadVariableOp�
AssignMovingAvg/sub_1Sub&AssignMovingAvg/ReadVariableOp:value:0FusedBatchNormV3:batch_mean:0*
T0**
_class 
loc:@AssignMovingAvg/4354742*
_output_shapes
: 2
AssignMovingAvg/sub_1�
AssignMovingAvg/mulMulAssignMovingAvg/sub_1:z:0AssignMovingAvg/sub:z:0*
T0**
_class 
loc:@AssignMovingAvg/4354742*
_output_shapes
: 2
AssignMovingAvg/mul�
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOpassignmovingavg_4354742AssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp**
_class 
loc:@AssignMovingAvg/4354742*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOp�
AssignMovingAvg_1/sub/xConst*,
_class"
 loc:@AssignMovingAvg_1/4354749*
_output_shapes
: *
dtype0*
valueB
 *  �?2
AssignMovingAvg_1/sub/x�
AssignMovingAvg_1/subSub AssignMovingAvg_1/sub/x:output:0Const_2:output:0*
T0*,
_class"
 loc:@AssignMovingAvg_1/4354749*
_output_shapes
: 2
AssignMovingAvg_1/sub�
 AssignMovingAvg_1/ReadVariableOpReadVariableOpassignmovingavg_1_4354749*
_output_shapes
: *
dtype02"
 AssignMovingAvg_1/ReadVariableOp�
AssignMovingAvg_1/sub_1Sub(AssignMovingAvg_1/ReadVariableOp:value:0!FusedBatchNormV3:batch_variance:0*
T0*,
_class"
 loc:@AssignMovingAvg_1/4354749*
_output_shapes
: 2
AssignMovingAvg_1/sub_1�
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub_1:z:0AssignMovingAvg_1/sub:z:0*
T0*,
_class"
 loc:@AssignMovingAvg_1/4354749*
_output_shapes
: 2
AssignMovingAvg_1/mul�
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOpassignmovingavg_1_4354749AssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*,
_class"
 loc:@AssignMovingAvg_1/4354749*
_output_shapes
 *
dtype02'
%AssignMovingAvg_1/AssignSubVariableOp�
IdentityIdentityFusedBatchNormV3:y:0$^AssignMovingAvg/AssignSubVariableOp^AssignMovingAvg/ReadVariableOp&^AssignMovingAvg_1/AssignSubVariableOp!^AssignMovingAvg_1/ReadVariableOp^ReadVariableOp^ReadVariableOp_1*
T0*/
_output_shapes
:���������   2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:���������   ::::2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp2D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp2 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:& "
 
_user_specified_nameinputs
�
�
;__inference_output_class_distribution_layer_call_fn_4355210

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*(
_output_shapes
:����������*-
config_proto

GPU

CPU2*0J 8*_
fZRX
V__inference_output_class_distribution_layer_call_and_return_conditional_losses_43534362
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:����������2

Identity"
identityIdentity:output:0*/
_input_shapes
:����������::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs
�
�
;__inference_input_batch_normalization_layer_call_fn_4354422

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4*
Tin	
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*1
_output_shapes
:�����������*-
config_proto

GPU

CPU2*0J 8*_
fZRX
V__inference_input_batch_normalization_layer_call_and_return_conditional_losses_43529142
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*1
_output_shapes
:�����������2

Identity"
identityIdentity:output:0*@
_input_shapes/
-:�����������::::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs
�
D
(__inference_RELU_2_layer_call_fn_4354807

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*/
_output_shapes
:���������   *-
config_proto

GPU

CPU2*0J 8*L
fGRE
C__inference_RELU_2_layer_call_and_return_conditional_losses_43531732
PartitionedCallt
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:���������   2

Identity"
identityIdentity:output:0*.
_input_shapes
:���������   :& "
 
_user_specified_nameinputs
�
�
E__inference_b_norm_4_layer_call_and_return_conditional_losses_4353336

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity��FusedBatchNormV3/ReadVariableOp�!FusedBatchNormV3/ReadVariableOp_1�ReadVariableOp�ReadVariableOp_1^
LogicalAnd/xConst*
_output_shapes
: *
dtype0
*
value	B
 Z 2
LogicalAnd/x^
LogicalAnd/yConst*
_output_shapes
: *
dtype0
*
value	B
 Z2
LogicalAnd/yl

LogicalAnd
LogicalAndLogicalAnd/x:output:0LogicalAnd/y:output:0*
_output_shapes
: 2

LogicalAndt
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOp�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:���������@:@:@:@:@:*
epsilon%o�:*
is_training( 2
FusedBatchNormV3S
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *�p}?2
Const�
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*/
_output_shapes
:���������@2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:���������@::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:& "
 
_user_specified_nameinputs
�
�
;__inference_input_batch_normalization_layer_call_fn_4354339

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4*
Tin	
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*A
_output_shapes/
-:+���������������������������*-
config_proto

GPU

CPU2*0J 8*_
fZRX
V__inference_input_batch_normalization_layer_call_and_return_conditional_losses_43521462
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+���������������������������2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+���������������������������::::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs
�
�
*__inference_b_norm_2_layer_call_fn_4354714

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4*
Tin	
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*A
_output_shapes/
-:+��������������������������� *-
config_proto

GPU

CPU2*0J 8*N
fIRG
E__inference_b_norm_2_layer_call_and_return_conditional_losses_43524742
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+��������������������������� 2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+��������������������������� ::::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs
�
�
*__inference_b_norm_2_layer_call_fn_4354797

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4*
Tin	
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*/
_output_shapes
:���������   *-
config_proto

GPU

CPU2*0J 8*N
fIRG
E__inference_b_norm_2_layer_call_and_return_conditional_losses_43531442
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:���������   2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:���������   ::::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs
�
�
*__inference_b_norm_4_layer_call_fn_4355063

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4*
Tin	
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*A
_output_shapes/
-:+���������������������������@*-
config_proto

GPU

CPU2*0J 8*N
fIRG
E__inference_b_norm_4_layer_call_and_return_conditional_losses_43528332
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+���������������������������@2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+���������������������������@::::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs
�$
�
E__inference_b_norm_4_layer_call_and_return_conditional_losses_4355097

inputs
readvariableop_resource
readvariableop_1_resource
assignmovingavg_4355082
assignmovingavg_1_4355089
identity��#AssignMovingAvg/AssignSubVariableOp�AssignMovingAvg/ReadVariableOp�%AssignMovingAvg_1/AssignSubVariableOp� AssignMovingAvg_1/ReadVariableOp�ReadVariableOp�ReadVariableOp_1^
LogicalAnd/xConst*
_output_shapes
: *
dtype0
*
value	B
 Z2
LogicalAnd/x^
LogicalAnd/yConst*
_output_shapes
: *
dtype0
*
value	B
 Z2
LogicalAnd/yl

LogicalAnd
LogicalAndLogicalAnd/x:output:0LogicalAnd/y:output:0*
_output_shapes
: 2

LogicalAndt
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1Q
ConstConst*
_output_shapes
: *
dtype0*
valueB 2
ConstU
Const_1Const*
_output_shapes
: *
dtype0*
valueB 2	
Const_1�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0Const:output:0Const_1:output:0*
T0*
U0*K
_output_shapes9
7:���������@:@:@:@:@:*
epsilon%o�:2
FusedBatchNormV3W
Const_2Const*
_output_shapes
: *
dtype0*
valueB
 *�p}?2	
Const_2�
AssignMovingAvg/sub/xConst**
_class 
loc:@AssignMovingAvg/4355082*
_output_shapes
: *
dtype0*
valueB
 *  �?2
AssignMovingAvg/sub/x�
AssignMovingAvg/subSubAssignMovingAvg/sub/x:output:0Const_2:output:0*
T0**
_class 
loc:@AssignMovingAvg/4355082*
_output_shapes
: 2
AssignMovingAvg/sub�
AssignMovingAvg/ReadVariableOpReadVariableOpassignmovingavg_4355082*
_output_shapes
:@*
dtype02 
AssignMovingAvg/ReadVariableOp�
AssignMovingAvg/sub_1Sub&AssignMovingAvg/ReadVariableOp:value:0FusedBatchNormV3:batch_mean:0*
T0**
_class 
loc:@AssignMovingAvg/4355082*
_output_shapes
:@2
AssignMovingAvg/sub_1�
AssignMovingAvg/mulMulAssignMovingAvg/sub_1:z:0AssignMovingAvg/sub:z:0*
T0**
_class 
loc:@AssignMovingAvg/4355082*
_output_shapes
:@2
AssignMovingAvg/mul�
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOpassignmovingavg_4355082AssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp**
_class 
loc:@AssignMovingAvg/4355082*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOp�
AssignMovingAvg_1/sub/xConst*,
_class"
 loc:@AssignMovingAvg_1/4355089*
_output_shapes
: *
dtype0*
valueB
 *  �?2
AssignMovingAvg_1/sub/x�
AssignMovingAvg_1/subSub AssignMovingAvg_1/sub/x:output:0Const_2:output:0*
T0*,
_class"
 loc:@AssignMovingAvg_1/4355089*
_output_shapes
: 2
AssignMovingAvg_1/sub�
 AssignMovingAvg_1/ReadVariableOpReadVariableOpassignmovingavg_1_4355089*
_output_shapes
:@*
dtype02"
 AssignMovingAvg_1/ReadVariableOp�
AssignMovingAvg_1/sub_1Sub(AssignMovingAvg_1/ReadVariableOp:value:0!FusedBatchNormV3:batch_variance:0*
T0*,
_class"
 loc:@AssignMovingAvg_1/4355089*
_output_shapes
:@2
AssignMovingAvg_1/sub_1�
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub_1:z:0AssignMovingAvg_1/sub:z:0*
T0*,
_class"
 loc:@AssignMovingAvg_1/4355089*
_output_shapes
:@2
AssignMovingAvg_1/mul�
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOpassignmovingavg_1_4355089AssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*,
_class"
 loc:@AssignMovingAvg_1/4355089*
_output_shapes
 *
dtype02'
%AssignMovingAvg_1/AssignSubVariableOp�
IdentityIdentityFusedBatchNormV3:y:0$^AssignMovingAvg/AssignSubVariableOp^AssignMovingAvg/ReadVariableOp&^AssignMovingAvg_1/AssignSubVariableOp!^AssignMovingAvg_1/ReadVariableOp^ReadVariableOp^ReadVariableOp_1*
T0*/
_output_shapes
:���������@2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:���������@::::2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp2D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp2 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:& "
 
_user_specified_nameinputs
�
�
*__inference_b_norm_2_layer_call_fn_4354788

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4*
Tin	
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*/
_output_shapes
:���������   *-
config_proto

GPU

CPU2*0J 8*N
fIRG
E__inference_b_norm_2_layer_call_and_return_conditional_losses_43531222
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:���������   2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:���������   ::::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs
�
e
G__inference_dropout_42_layer_call_and_return_conditional_losses_4354457

inputs

identity_1b
IdentityIdentityinputs*
T0*/
_output_shapes
:���������@@2

Identityq

Identity_1IdentityIdentity:output:0*
T0*/
_output_shapes
:���������@@2

Identity_1"!

identity_1Identity_1:output:0*.
_input_shapes
:���������@@:& "
 
_user_specified_nameinputs
�$
�
E__inference_b_norm_1_layer_call_and_return_conditional_losses_4353026

inputs
readvariableop_resource
readvariableop_1_resource
assignmovingavg_4353011
assignmovingavg_1_4353018
identity��#AssignMovingAvg/AssignSubVariableOp�AssignMovingAvg/ReadVariableOp�%AssignMovingAvg_1/AssignSubVariableOp� AssignMovingAvg_1/ReadVariableOp�ReadVariableOp�ReadVariableOp_1^
LogicalAnd/xConst*
_output_shapes
: *
dtype0
*
value	B
 Z2
LogicalAnd/x^
LogicalAnd/yConst*
_output_shapes
: *
dtype0
*
value	B
 Z2
LogicalAnd/yl

LogicalAnd
LogicalAndLogicalAnd/x:output:0LogicalAnd/y:output:0*
_output_shapes
: 2

LogicalAndt
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1Q
ConstConst*
_output_shapes
: *
dtype0*
valueB 2
ConstU
Const_1Const*
_output_shapes
: *
dtype0*
valueB 2	
Const_1�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0Const:output:0Const_1:output:0*
T0*
U0*K
_output_shapes9
7:���������@@:::::*
epsilon%o�:2
FusedBatchNormV3W
Const_2Const*
_output_shapes
: *
dtype0*
valueB
 *�p}?2	
Const_2�
AssignMovingAvg/sub/xConst**
_class 
loc:@AssignMovingAvg/4353011*
_output_shapes
: *
dtype0*
valueB
 *  �?2
AssignMovingAvg/sub/x�
AssignMovingAvg/subSubAssignMovingAvg/sub/x:output:0Const_2:output:0*
T0**
_class 
loc:@AssignMovingAvg/4353011*
_output_shapes
: 2
AssignMovingAvg/sub�
AssignMovingAvg/ReadVariableOpReadVariableOpassignmovingavg_4353011*
_output_shapes
:*
dtype02 
AssignMovingAvg/ReadVariableOp�
AssignMovingAvg/sub_1Sub&AssignMovingAvg/ReadVariableOp:value:0FusedBatchNormV3:batch_mean:0*
T0**
_class 
loc:@AssignMovingAvg/4353011*
_output_shapes
:2
AssignMovingAvg/sub_1�
AssignMovingAvg/mulMulAssignMovingAvg/sub_1:z:0AssignMovingAvg/sub:z:0*
T0**
_class 
loc:@AssignMovingAvg/4353011*
_output_shapes
:2
AssignMovingAvg/mul�
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOpassignmovingavg_4353011AssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp**
_class 
loc:@AssignMovingAvg/4353011*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOp�
AssignMovingAvg_1/sub/xConst*,
_class"
 loc:@AssignMovingAvg_1/4353018*
_output_shapes
: *
dtype0*
valueB
 *  �?2
AssignMovingAvg_1/sub/x�
AssignMovingAvg_1/subSub AssignMovingAvg_1/sub/x:output:0Const_2:output:0*
T0*,
_class"
 loc:@AssignMovingAvg_1/4353018*
_output_shapes
: 2
AssignMovingAvg_1/sub�
 AssignMovingAvg_1/ReadVariableOpReadVariableOpassignmovingavg_1_4353018*
_output_shapes
:*
dtype02"
 AssignMovingAvg_1/ReadVariableOp�
AssignMovingAvg_1/sub_1Sub(AssignMovingAvg_1/ReadVariableOp:value:0!FusedBatchNormV3:batch_variance:0*
T0*,
_class"
 loc:@AssignMovingAvg_1/4353018*
_output_shapes
:2
AssignMovingAvg_1/sub_1�
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub_1:z:0AssignMovingAvg_1/sub:z:0*
T0*,
_class"
 loc:@AssignMovingAvg_1/4353018*
_output_shapes
:2
AssignMovingAvg_1/mul�
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOpassignmovingavg_1_4353018AssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*,
_class"
 loc:@AssignMovingAvg_1/4353018*
_output_shapes
 *
dtype02'
%AssignMovingAvg_1/AssignSubVariableOp�
IdentityIdentityFusedBatchNormV3:y:0$^AssignMovingAvg/AssignSubVariableOp^AssignMovingAvg/ReadVariableOp&^AssignMovingAvg_1/AssignSubVariableOp!^AssignMovingAvg_1/ReadVariableOp^ReadVariableOp^ReadVariableOp_1*
T0*/
_output_shapes
:���������@@2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:���������@@::::2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp2D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp2 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:& "
 
_user_specified_nameinputs
�
N
2__inference_max_pooling2d_84_layer_call_fn_4352852

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*J
_output_shapes8
6:4������������������������������������*-
config_proto

GPU

CPU2*0J 8*V
fQRO
M__inference_max_pooling2d_84_layer_call_and_return_conditional_losses_43528462
PartitionedCall�
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4������������������������������������2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4������������������������������������:& "
 
_user_specified_nameinputs
�
�
E__inference_b_norm_1_layer_call_and_return_conditional_losses_4354609

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity��FusedBatchNormV3/ReadVariableOp�!FusedBatchNormV3/ReadVariableOp_1�ReadVariableOp�ReadVariableOp_1^
LogicalAnd/xConst*
_output_shapes
: *
dtype0
*
value	B
 Z 2
LogicalAnd/x^
LogicalAnd/yConst*
_output_shapes
: *
dtype0
*
value	B
 Z2
LogicalAnd/yl

LogicalAnd
LogicalAndLogicalAnd/x:output:0LogicalAnd/y:output:0*
_output_shapes
: 2

LogicalAndt
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOp�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+���������������������������:::::*
epsilon%o�:*
is_training( 2
FusedBatchNormV3S
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *�p}?2
Const�
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*A
_output_shapes/
-:+���������������������������2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+���������������������������::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:& "
 
_user_specified_nameinputs
�
�
E__inference_b_norm_1_layer_call_and_return_conditional_losses_4354535

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity��FusedBatchNormV3/ReadVariableOp�!FusedBatchNormV3/ReadVariableOp_1�ReadVariableOp�ReadVariableOp_1^
LogicalAnd/xConst*
_output_shapes
: *
dtype0
*
value	B
 Z 2
LogicalAnd/x^
LogicalAnd/yConst*
_output_shapes
: *
dtype0
*
value	B
 Z2
LogicalAnd/yl

LogicalAnd
LogicalAndLogicalAnd/x:output:0LogicalAnd/y:output:0*
_output_shapes
: 2

LogicalAndt
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOp�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:���������@@:::::*
epsilon%o�:*
is_training( 2
FusedBatchNormV3S
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *�p}?2
Const�
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*/
_output_shapes
:���������@@2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:���������@@::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:& "
 
_user_specified_nameinputs
�
N
2__inference_max_pooling2d_83_layer_call_fn_4352688

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*J
_output_shapes8
6:4������������������������������������*-
config_proto

GPU

CPU2*0J 8*V
fQRO
M__inference_max_pooling2d_83_layer_call_and_return_conditional_losses_43526822
PartitionedCall�
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4������������������������������������2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4������������������������������������:& "
 
_user_specified_nameinputs
��
�
"__inference__wrapped_model_4352032
input_conv_input8
4awsome_net_input_conv_conv2d_readvariableop_resource9
5awsome_net_input_conv_biasadd_readvariableop_resource@
<awsome_net_input_batch_normalization_readvariableop_resourceB
>awsome_net_input_batch_normalization_readvariableop_1_resourceQ
Mawsome_net_input_batch_normalization_fusedbatchnormv3_readvariableop_resourceS
Oawsome_net_input_batch_normalization_fusedbatchnormv3_readvariableop_1_resource4
0awsome_net_conv_1_conv2d_readvariableop_resource5
1awsome_net_conv_1_biasadd_readvariableop_resource/
+awsome_net_b_norm_1_readvariableop_resource1
-awsome_net_b_norm_1_readvariableop_1_resource@
<awsome_net_b_norm_1_fusedbatchnormv3_readvariableop_resourceB
>awsome_net_b_norm_1_fusedbatchnormv3_readvariableop_1_resource4
0awsome_net_conv_2_conv2d_readvariableop_resource5
1awsome_net_conv_2_biasadd_readvariableop_resource/
+awsome_net_b_norm_2_readvariableop_resource1
-awsome_net_b_norm_2_readvariableop_1_resource@
<awsome_net_b_norm_2_fusedbatchnormv3_readvariableop_resourceB
>awsome_net_b_norm_2_fusedbatchnormv3_readvariableop_1_resource4
0awsome_net_conv_3_conv2d_readvariableop_resource5
1awsome_net_conv_3_biasadd_readvariableop_resource/
+awsome_net_b_norm_3_readvariableop_resource1
-awsome_net_b_norm_3_readvariableop_1_resource@
<awsome_net_b_norm_3_fusedbatchnormv3_readvariableop_resourceB
>awsome_net_b_norm_3_fusedbatchnormv3_readvariableop_1_resource4
0awsome_net_conv_4_conv2d_readvariableop_resource5
1awsome_net_conv_4_biasadd_readvariableop_resource/
+awsome_net_b_norm_4_readvariableop_resource1
-awsome_net_b_norm_4_readvariableop_1_resource@
<awsome_net_b_norm_4_fusedbatchnormv3_readvariableop_resourceB
>awsome_net_b_norm_4_fusedbatchnormv3_readvariableop_1_resourceG
Cawsome_net_output_class_distribution_matmul_readvariableop_resourceH
Dawsome_net_output_class_distribution_biasadd_readvariableop_resource
identity��3awsome_net/b_norm_1/FusedBatchNormV3/ReadVariableOp�5awsome_net/b_norm_1/FusedBatchNormV3/ReadVariableOp_1�"awsome_net/b_norm_1/ReadVariableOp�$awsome_net/b_norm_1/ReadVariableOp_1�3awsome_net/b_norm_2/FusedBatchNormV3/ReadVariableOp�5awsome_net/b_norm_2/FusedBatchNormV3/ReadVariableOp_1�"awsome_net/b_norm_2/ReadVariableOp�$awsome_net/b_norm_2/ReadVariableOp_1�3awsome_net/b_norm_3/FusedBatchNormV3/ReadVariableOp�5awsome_net/b_norm_3/FusedBatchNormV3/ReadVariableOp_1�"awsome_net/b_norm_3/ReadVariableOp�$awsome_net/b_norm_3/ReadVariableOp_1�3awsome_net/b_norm_4/FusedBatchNormV3/ReadVariableOp�5awsome_net/b_norm_4/FusedBatchNormV3/ReadVariableOp_1�"awsome_net/b_norm_4/ReadVariableOp�$awsome_net/b_norm_4/ReadVariableOp_1�(awsome_net/conv_1/BiasAdd/ReadVariableOp�'awsome_net/conv_1/Conv2D/ReadVariableOp�(awsome_net/conv_2/BiasAdd/ReadVariableOp�'awsome_net/conv_2/Conv2D/ReadVariableOp�(awsome_net/conv_3/BiasAdd/ReadVariableOp�'awsome_net/conv_3/Conv2D/ReadVariableOp�(awsome_net/conv_4/BiasAdd/ReadVariableOp�'awsome_net/conv_4/Conv2D/ReadVariableOp�Dawsome_net/input_batch_normalization/FusedBatchNormV3/ReadVariableOp�Fawsome_net/input_batch_normalization/FusedBatchNormV3/ReadVariableOp_1�3awsome_net/input_batch_normalization/ReadVariableOp�5awsome_net/input_batch_normalization/ReadVariableOp_1�,awsome_net/input_conv/BiasAdd/ReadVariableOp�+awsome_net/input_conv/Conv2D/ReadVariableOp�;awsome_net/output_class_distribution/BiasAdd/ReadVariableOp�:awsome_net/output_class_distribution/MatMul/ReadVariableOp�
+awsome_net/input_conv/Conv2D/ReadVariableOpReadVariableOp4awsome_net_input_conv_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02-
+awsome_net/input_conv/Conv2D/ReadVariableOp�
awsome_net/input_conv/Conv2DConv2Dinput_conv_input3awsome_net/input_conv/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:�����������*
paddingSAME*
strides
2
awsome_net/input_conv/Conv2D�
,awsome_net/input_conv/BiasAdd/ReadVariableOpReadVariableOp5awsome_net_input_conv_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02.
,awsome_net/input_conv/BiasAdd/ReadVariableOp�
awsome_net/input_conv/BiasAddBiasAdd%awsome_net/input_conv/Conv2D:output:04awsome_net/input_conv/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:�����������2
awsome_net/input_conv/BiasAdd�
1awsome_net/input_batch_normalization/LogicalAnd/xConst*
_output_shapes
: *
dtype0
*
value	B
 Z 23
1awsome_net/input_batch_normalization/LogicalAnd/x�
1awsome_net/input_batch_normalization/LogicalAnd/yConst*
_output_shapes
: *
dtype0
*
value	B
 Z23
1awsome_net/input_batch_normalization/LogicalAnd/y�
/awsome_net/input_batch_normalization/LogicalAnd
LogicalAnd:awsome_net/input_batch_normalization/LogicalAnd/x:output:0:awsome_net/input_batch_normalization/LogicalAnd/y:output:0*
_output_shapes
: 21
/awsome_net/input_batch_normalization/LogicalAnd�
3awsome_net/input_batch_normalization/ReadVariableOpReadVariableOp<awsome_net_input_batch_normalization_readvariableop_resource*
_output_shapes
:*
dtype025
3awsome_net/input_batch_normalization/ReadVariableOp�
5awsome_net/input_batch_normalization/ReadVariableOp_1ReadVariableOp>awsome_net_input_batch_normalization_readvariableop_1_resource*
_output_shapes
:*
dtype027
5awsome_net/input_batch_normalization/ReadVariableOp_1�
Dawsome_net/input_batch_normalization/FusedBatchNormV3/ReadVariableOpReadVariableOpMawsome_net_input_batch_normalization_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02F
Dawsome_net/input_batch_normalization/FusedBatchNormV3/ReadVariableOp�
Fawsome_net/input_batch_normalization/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpOawsome_net_input_batch_normalization_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02H
Fawsome_net/input_batch_normalization/FusedBatchNormV3/ReadVariableOp_1�
5awsome_net/input_batch_normalization/FusedBatchNormV3FusedBatchNormV3&awsome_net/input_conv/BiasAdd:output:0;awsome_net/input_batch_normalization/ReadVariableOp:value:0=awsome_net/input_batch_normalization/ReadVariableOp_1:value:0Lawsome_net/input_batch_normalization/FusedBatchNormV3/ReadVariableOp:value:0Nawsome_net/input_batch_normalization/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*M
_output_shapes;
9:�����������:::::*
epsilon%o�:*
is_training( 27
5awsome_net/input_batch_normalization/FusedBatchNormV3�
*awsome_net/input_batch_normalization/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *�p}?2,
*awsome_net/input_batch_normalization/Const�
awsome_net/input_RELU/ReluRelu9awsome_net/input_batch_normalization/FusedBatchNormV3:y:0*
T0*1
_output_shapes
:�����������2
awsome_net/input_RELU/Relu�
#awsome_net/max_pooling2d_85/MaxPoolMaxPool(awsome_net/input_RELU/Relu:activations:0*/
_output_shapes
:���������@@*
ksize
*
paddingVALID*
strides
2%
#awsome_net/max_pooling2d_85/MaxPool�
awsome_net/dropout_42/IdentityIdentity,awsome_net/max_pooling2d_85/MaxPool:output:0*
T0*/
_output_shapes
:���������@@2 
awsome_net/dropout_42/Identity�
'awsome_net/conv_1/Conv2D/ReadVariableOpReadVariableOp0awsome_net_conv_1_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02)
'awsome_net/conv_1/Conv2D/ReadVariableOp�
awsome_net/conv_1/Conv2DConv2D'awsome_net/dropout_42/Identity:output:0/awsome_net/conv_1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@@*
paddingSAME*
strides
2
awsome_net/conv_1/Conv2D�
(awsome_net/conv_1/BiasAdd/ReadVariableOpReadVariableOp1awsome_net_conv_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02*
(awsome_net/conv_1/BiasAdd/ReadVariableOp�
awsome_net/conv_1/BiasAddBiasAdd!awsome_net/conv_1/Conv2D:output:00awsome_net/conv_1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@@2
awsome_net/conv_1/BiasAdd�
 awsome_net/b_norm_1/LogicalAnd/xConst*
_output_shapes
: *
dtype0
*
value	B
 Z 2"
 awsome_net/b_norm_1/LogicalAnd/x�
 awsome_net/b_norm_1/LogicalAnd/yConst*
_output_shapes
: *
dtype0
*
value	B
 Z2"
 awsome_net/b_norm_1/LogicalAnd/y�
awsome_net/b_norm_1/LogicalAnd
LogicalAnd)awsome_net/b_norm_1/LogicalAnd/x:output:0)awsome_net/b_norm_1/LogicalAnd/y:output:0*
_output_shapes
: 2 
awsome_net/b_norm_1/LogicalAnd�
"awsome_net/b_norm_1/ReadVariableOpReadVariableOp+awsome_net_b_norm_1_readvariableop_resource*
_output_shapes
:*
dtype02$
"awsome_net/b_norm_1/ReadVariableOp�
$awsome_net/b_norm_1/ReadVariableOp_1ReadVariableOp-awsome_net_b_norm_1_readvariableop_1_resource*
_output_shapes
:*
dtype02&
$awsome_net/b_norm_1/ReadVariableOp_1�
3awsome_net/b_norm_1/FusedBatchNormV3/ReadVariableOpReadVariableOp<awsome_net_b_norm_1_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype025
3awsome_net/b_norm_1/FusedBatchNormV3/ReadVariableOp�
5awsome_net/b_norm_1/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp>awsome_net_b_norm_1_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype027
5awsome_net/b_norm_1/FusedBatchNormV3/ReadVariableOp_1�
$awsome_net/b_norm_1/FusedBatchNormV3FusedBatchNormV3"awsome_net/conv_1/BiasAdd:output:0*awsome_net/b_norm_1/ReadVariableOp:value:0,awsome_net/b_norm_1/ReadVariableOp_1:value:0;awsome_net/b_norm_1/FusedBatchNormV3/ReadVariableOp:value:0=awsome_net/b_norm_1/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:���������@@:::::*
epsilon%o�:*
is_training( 2&
$awsome_net/b_norm_1/FusedBatchNormV3{
awsome_net/b_norm_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *�p}?2
awsome_net/b_norm_1/Const�
awsome_net/RELU_1/ReluRelu(awsome_net/b_norm_1/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:���������@@2
awsome_net/RELU_1/Relu�
#awsome_net/max_pooling2d_81/MaxPoolMaxPool$awsome_net/RELU_1/Relu:activations:0*/
_output_shapes
:���������  *
ksize
*
paddingVALID*
strides
2%
#awsome_net/max_pooling2d_81/MaxPool�
'awsome_net/conv_2/Conv2D/ReadVariableOpReadVariableOp0awsome_net_conv_2_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02)
'awsome_net/conv_2/Conv2D/ReadVariableOp�
awsome_net/conv_2/Conv2DConv2D,awsome_net/max_pooling2d_81/MaxPool:output:0/awsome_net/conv_2/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������   *
paddingSAME*
strides
2
awsome_net/conv_2/Conv2D�
(awsome_net/conv_2/BiasAdd/ReadVariableOpReadVariableOp1awsome_net_conv_2_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02*
(awsome_net/conv_2/BiasAdd/ReadVariableOp�
awsome_net/conv_2/BiasAddBiasAdd!awsome_net/conv_2/Conv2D:output:00awsome_net/conv_2/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������   2
awsome_net/conv_2/BiasAdd�
 awsome_net/b_norm_2/LogicalAnd/xConst*
_output_shapes
: *
dtype0
*
value	B
 Z 2"
 awsome_net/b_norm_2/LogicalAnd/x�
 awsome_net/b_norm_2/LogicalAnd/yConst*
_output_shapes
: *
dtype0
*
value	B
 Z2"
 awsome_net/b_norm_2/LogicalAnd/y�
awsome_net/b_norm_2/LogicalAnd
LogicalAnd)awsome_net/b_norm_2/LogicalAnd/x:output:0)awsome_net/b_norm_2/LogicalAnd/y:output:0*
_output_shapes
: 2 
awsome_net/b_norm_2/LogicalAnd�
"awsome_net/b_norm_2/ReadVariableOpReadVariableOp+awsome_net_b_norm_2_readvariableop_resource*
_output_shapes
: *
dtype02$
"awsome_net/b_norm_2/ReadVariableOp�
$awsome_net/b_norm_2/ReadVariableOp_1ReadVariableOp-awsome_net_b_norm_2_readvariableop_1_resource*
_output_shapes
: *
dtype02&
$awsome_net/b_norm_2/ReadVariableOp_1�
3awsome_net/b_norm_2/FusedBatchNormV3/ReadVariableOpReadVariableOp<awsome_net_b_norm_2_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype025
3awsome_net/b_norm_2/FusedBatchNormV3/ReadVariableOp�
5awsome_net/b_norm_2/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp>awsome_net_b_norm_2_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype027
5awsome_net/b_norm_2/FusedBatchNormV3/ReadVariableOp_1�
$awsome_net/b_norm_2/FusedBatchNormV3FusedBatchNormV3"awsome_net/conv_2/BiasAdd:output:0*awsome_net/b_norm_2/ReadVariableOp:value:0,awsome_net/b_norm_2/ReadVariableOp_1:value:0;awsome_net/b_norm_2/FusedBatchNormV3/ReadVariableOp:value:0=awsome_net/b_norm_2/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:���������   : : : : :*
epsilon%o�:*
is_training( 2&
$awsome_net/b_norm_2/FusedBatchNormV3{
awsome_net/b_norm_2/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *�p}?2
awsome_net/b_norm_2/Const�
awsome_net/RELU_2/ReluRelu(awsome_net/b_norm_2/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:���������   2
awsome_net/RELU_2/Relu�
#awsome_net/max_pooling2d_82/MaxPoolMaxPool$awsome_net/RELU_2/Relu:activations:0*/
_output_shapes
:��������� *
ksize
*
paddingVALID*
strides
2%
#awsome_net/max_pooling2d_82/MaxPool�
'awsome_net/conv_3/Conv2D/ReadVariableOpReadVariableOp0awsome_net_conv_3_conv2d_readvariableop_resource*&
_output_shapes
: @*
dtype02)
'awsome_net/conv_3/Conv2D/ReadVariableOp�
awsome_net/conv_3/Conv2DConv2D,awsome_net/max_pooling2d_82/MaxPool:output:0/awsome_net/conv_3/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@*
paddingSAME*
strides
2
awsome_net/conv_3/Conv2D�
(awsome_net/conv_3/BiasAdd/ReadVariableOpReadVariableOp1awsome_net_conv_3_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02*
(awsome_net/conv_3/BiasAdd/ReadVariableOp�
awsome_net/conv_3/BiasAddBiasAdd!awsome_net/conv_3/Conv2D:output:00awsome_net/conv_3/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@2
awsome_net/conv_3/BiasAdd�
 awsome_net/b_norm_3/LogicalAnd/xConst*
_output_shapes
: *
dtype0
*
value	B
 Z 2"
 awsome_net/b_norm_3/LogicalAnd/x�
 awsome_net/b_norm_3/LogicalAnd/yConst*
_output_shapes
: *
dtype0
*
value	B
 Z2"
 awsome_net/b_norm_3/LogicalAnd/y�
awsome_net/b_norm_3/LogicalAnd
LogicalAnd)awsome_net/b_norm_3/LogicalAnd/x:output:0)awsome_net/b_norm_3/LogicalAnd/y:output:0*
_output_shapes
: 2 
awsome_net/b_norm_3/LogicalAnd�
"awsome_net/b_norm_3/ReadVariableOpReadVariableOp+awsome_net_b_norm_3_readvariableop_resource*
_output_shapes
:@*
dtype02$
"awsome_net/b_norm_3/ReadVariableOp�
$awsome_net/b_norm_3/ReadVariableOp_1ReadVariableOp-awsome_net_b_norm_3_readvariableop_1_resource*
_output_shapes
:@*
dtype02&
$awsome_net/b_norm_3/ReadVariableOp_1�
3awsome_net/b_norm_3/FusedBatchNormV3/ReadVariableOpReadVariableOp<awsome_net_b_norm_3_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype025
3awsome_net/b_norm_3/FusedBatchNormV3/ReadVariableOp�
5awsome_net/b_norm_3/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp>awsome_net_b_norm_3_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype027
5awsome_net/b_norm_3/FusedBatchNormV3/ReadVariableOp_1�
$awsome_net/b_norm_3/FusedBatchNormV3FusedBatchNormV3"awsome_net/conv_3/BiasAdd:output:0*awsome_net/b_norm_3/ReadVariableOp:value:0,awsome_net/b_norm_3/ReadVariableOp_1:value:0;awsome_net/b_norm_3/FusedBatchNormV3/ReadVariableOp:value:0=awsome_net/b_norm_3/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:���������@:@:@:@:@:*
epsilon%o�:*
is_training( 2&
$awsome_net/b_norm_3/FusedBatchNormV3{
awsome_net/b_norm_3/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *�p}?2
awsome_net/b_norm_3/Const�
awsome_net/RELU_3/ReluRelu(awsome_net/b_norm_3/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:���������@2
awsome_net/RELU_3/Relu�
#awsome_net/max_pooling2d_83/MaxPoolMaxPool$awsome_net/RELU_3/Relu:activations:0*/
_output_shapes
:���������@*
ksize
*
paddingVALID*
strides
2%
#awsome_net/max_pooling2d_83/MaxPool�
'awsome_net/conv_4/Conv2D/ReadVariableOpReadVariableOp0awsome_net_conv_4_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02)
'awsome_net/conv_4/Conv2D/ReadVariableOp�
awsome_net/conv_4/Conv2DConv2D,awsome_net/max_pooling2d_83/MaxPool:output:0/awsome_net/conv_4/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@*
paddingSAME*
strides
2
awsome_net/conv_4/Conv2D�
(awsome_net/conv_4/BiasAdd/ReadVariableOpReadVariableOp1awsome_net_conv_4_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02*
(awsome_net/conv_4/BiasAdd/ReadVariableOp�
awsome_net/conv_4/BiasAddBiasAdd!awsome_net/conv_4/Conv2D:output:00awsome_net/conv_4/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@2
awsome_net/conv_4/BiasAdd�
 awsome_net/b_norm_4/LogicalAnd/xConst*
_output_shapes
: *
dtype0
*
value	B
 Z 2"
 awsome_net/b_norm_4/LogicalAnd/x�
 awsome_net/b_norm_4/LogicalAnd/yConst*
_output_shapes
: *
dtype0
*
value	B
 Z2"
 awsome_net/b_norm_4/LogicalAnd/y�
awsome_net/b_norm_4/LogicalAnd
LogicalAnd)awsome_net/b_norm_4/LogicalAnd/x:output:0)awsome_net/b_norm_4/LogicalAnd/y:output:0*
_output_shapes
: 2 
awsome_net/b_norm_4/LogicalAnd�
"awsome_net/b_norm_4/ReadVariableOpReadVariableOp+awsome_net_b_norm_4_readvariableop_resource*
_output_shapes
:@*
dtype02$
"awsome_net/b_norm_4/ReadVariableOp�
$awsome_net/b_norm_4/ReadVariableOp_1ReadVariableOp-awsome_net_b_norm_4_readvariableop_1_resource*
_output_shapes
:@*
dtype02&
$awsome_net/b_norm_4/ReadVariableOp_1�
3awsome_net/b_norm_4/FusedBatchNormV3/ReadVariableOpReadVariableOp<awsome_net_b_norm_4_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype025
3awsome_net/b_norm_4/FusedBatchNormV3/ReadVariableOp�
5awsome_net/b_norm_4/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp>awsome_net_b_norm_4_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype027
5awsome_net/b_norm_4/FusedBatchNormV3/ReadVariableOp_1�
$awsome_net/b_norm_4/FusedBatchNormV3FusedBatchNormV3"awsome_net/conv_4/BiasAdd:output:0*awsome_net/b_norm_4/ReadVariableOp:value:0,awsome_net/b_norm_4/ReadVariableOp_1:value:0;awsome_net/b_norm_4/FusedBatchNormV3/ReadVariableOp:value:0=awsome_net/b_norm_4/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:���������@:@:@:@:@:*
epsilon%o�:*
is_training( 2&
$awsome_net/b_norm_4/FusedBatchNormV3{
awsome_net/b_norm_4/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *�p}?2
awsome_net/b_norm_4/Const�
awsome_net/RELU_4/ReluRelu(awsome_net/b_norm_4/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:���������@2
awsome_net/RELU_4/Relu�
#awsome_net/max_pooling2d_84/MaxPoolMaxPool$awsome_net/RELU_4/Relu:activations:0*/
_output_shapes
:���������@*
ksize
*
paddingVALID*
strides
2%
#awsome_net/max_pooling2d_84/MaxPool�
awsome_net/flatten_21/ConstConst*
_output_shapes
:*
dtype0*
valueB"����   2
awsome_net/flatten_21/Const�
awsome_net/flatten_21/ReshapeReshape,awsome_net/max_pooling2d_84/MaxPool:output:0$awsome_net/flatten_21/Const:output:0*
T0*(
_output_shapes
:����������2
awsome_net/flatten_21/Reshape�
awsome_net/dropout_43/IdentityIdentity&awsome_net/flatten_21/Reshape:output:0*
T0*(
_output_shapes
:����������2 
awsome_net/dropout_43/Identity�
:awsome_net/output_class_distribution/MatMul/ReadVariableOpReadVariableOpCawsome_net_output_class_distribution_matmul_readvariableop_resource* 
_output_shapes
:
��*
dtype02<
:awsome_net/output_class_distribution/MatMul/ReadVariableOp�
+awsome_net/output_class_distribution/MatMulMatMul'awsome_net/dropout_43/Identity:output:0Bawsome_net/output_class_distribution/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2-
+awsome_net/output_class_distribution/MatMul�
;awsome_net/output_class_distribution/BiasAdd/ReadVariableOpReadVariableOpDawsome_net_output_class_distribution_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02=
;awsome_net/output_class_distribution/BiasAdd/ReadVariableOp�
,awsome_net/output_class_distribution/BiasAddBiasAdd5awsome_net/output_class_distribution/MatMul:product:0Cawsome_net/output_class_distribution/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2.
,awsome_net/output_class_distribution/BiasAdd�
IdentityIdentity5awsome_net/output_class_distribution/BiasAdd:output:04^awsome_net/b_norm_1/FusedBatchNormV3/ReadVariableOp6^awsome_net/b_norm_1/FusedBatchNormV3/ReadVariableOp_1#^awsome_net/b_norm_1/ReadVariableOp%^awsome_net/b_norm_1/ReadVariableOp_14^awsome_net/b_norm_2/FusedBatchNormV3/ReadVariableOp6^awsome_net/b_norm_2/FusedBatchNormV3/ReadVariableOp_1#^awsome_net/b_norm_2/ReadVariableOp%^awsome_net/b_norm_2/ReadVariableOp_14^awsome_net/b_norm_3/FusedBatchNormV3/ReadVariableOp6^awsome_net/b_norm_3/FusedBatchNormV3/ReadVariableOp_1#^awsome_net/b_norm_3/ReadVariableOp%^awsome_net/b_norm_3/ReadVariableOp_14^awsome_net/b_norm_4/FusedBatchNormV3/ReadVariableOp6^awsome_net/b_norm_4/FusedBatchNormV3/ReadVariableOp_1#^awsome_net/b_norm_4/ReadVariableOp%^awsome_net/b_norm_4/ReadVariableOp_1)^awsome_net/conv_1/BiasAdd/ReadVariableOp(^awsome_net/conv_1/Conv2D/ReadVariableOp)^awsome_net/conv_2/BiasAdd/ReadVariableOp(^awsome_net/conv_2/Conv2D/ReadVariableOp)^awsome_net/conv_3/BiasAdd/ReadVariableOp(^awsome_net/conv_3/Conv2D/ReadVariableOp)^awsome_net/conv_4/BiasAdd/ReadVariableOp(^awsome_net/conv_4/Conv2D/ReadVariableOpE^awsome_net/input_batch_normalization/FusedBatchNormV3/ReadVariableOpG^awsome_net/input_batch_normalization/FusedBatchNormV3/ReadVariableOp_14^awsome_net/input_batch_normalization/ReadVariableOp6^awsome_net/input_batch_normalization/ReadVariableOp_1-^awsome_net/input_conv/BiasAdd/ReadVariableOp,^awsome_net/input_conv/Conv2D/ReadVariableOp<^awsome_net/output_class_distribution/BiasAdd/ReadVariableOp;^awsome_net/output_class_distribution/MatMul/ReadVariableOp*
T0*(
_output_shapes
:����������2

Identity"
identityIdentity:output:0*�
_input_shapes�
�:�����������::::::::::::::::::::::::::::::::2j
3awsome_net/b_norm_1/FusedBatchNormV3/ReadVariableOp3awsome_net/b_norm_1/FusedBatchNormV3/ReadVariableOp2n
5awsome_net/b_norm_1/FusedBatchNormV3/ReadVariableOp_15awsome_net/b_norm_1/FusedBatchNormV3/ReadVariableOp_12H
"awsome_net/b_norm_1/ReadVariableOp"awsome_net/b_norm_1/ReadVariableOp2L
$awsome_net/b_norm_1/ReadVariableOp_1$awsome_net/b_norm_1/ReadVariableOp_12j
3awsome_net/b_norm_2/FusedBatchNormV3/ReadVariableOp3awsome_net/b_norm_2/FusedBatchNormV3/ReadVariableOp2n
5awsome_net/b_norm_2/FusedBatchNormV3/ReadVariableOp_15awsome_net/b_norm_2/FusedBatchNormV3/ReadVariableOp_12H
"awsome_net/b_norm_2/ReadVariableOp"awsome_net/b_norm_2/ReadVariableOp2L
$awsome_net/b_norm_2/ReadVariableOp_1$awsome_net/b_norm_2/ReadVariableOp_12j
3awsome_net/b_norm_3/FusedBatchNormV3/ReadVariableOp3awsome_net/b_norm_3/FusedBatchNormV3/ReadVariableOp2n
5awsome_net/b_norm_3/FusedBatchNormV3/ReadVariableOp_15awsome_net/b_norm_3/FusedBatchNormV3/ReadVariableOp_12H
"awsome_net/b_norm_3/ReadVariableOp"awsome_net/b_norm_3/ReadVariableOp2L
$awsome_net/b_norm_3/ReadVariableOp_1$awsome_net/b_norm_3/ReadVariableOp_12j
3awsome_net/b_norm_4/FusedBatchNormV3/ReadVariableOp3awsome_net/b_norm_4/FusedBatchNormV3/ReadVariableOp2n
5awsome_net/b_norm_4/FusedBatchNormV3/ReadVariableOp_15awsome_net/b_norm_4/FusedBatchNormV3/ReadVariableOp_12H
"awsome_net/b_norm_4/ReadVariableOp"awsome_net/b_norm_4/ReadVariableOp2L
$awsome_net/b_norm_4/ReadVariableOp_1$awsome_net/b_norm_4/ReadVariableOp_12T
(awsome_net/conv_1/BiasAdd/ReadVariableOp(awsome_net/conv_1/BiasAdd/ReadVariableOp2R
'awsome_net/conv_1/Conv2D/ReadVariableOp'awsome_net/conv_1/Conv2D/ReadVariableOp2T
(awsome_net/conv_2/BiasAdd/ReadVariableOp(awsome_net/conv_2/BiasAdd/ReadVariableOp2R
'awsome_net/conv_2/Conv2D/ReadVariableOp'awsome_net/conv_2/Conv2D/ReadVariableOp2T
(awsome_net/conv_3/BiasAdd/ReadVariableOp(awsome_net/conv_3/BiasAdd/ReadVariableOp2R
'awsome_net/conv_3/Conv2D/ReadVariableOp'awsome_net/conv_3/Conv2D/ReadVariableOp2T
(awsome_net/conv_4/BiasAdd/ReadVariableOp(awsome_net/conv_4/BiasAdd/ReadVariableOp2R
'awsome_net/conv_4/Conv2D/ReadVariableOp'awsome_net/conv_4/Conv2D/ReadVariableOp2�
Dawsome_net/input_batch_normalization/FusedBatchNormV3/ReadVariableOpDawsome_net/input_batch_normalization/FusedBatchNormV3/ReadVariableOp2�
Fawsome_net/input_batch_normalization/FusedBatchNormV3/ReadVariableOp_1Fawsome_net/input_batch_normalization/FusedBatchNormV3/ReadVariableOp_12j
3awsome_net/input_batch_normalization/ReadVariableOp3awsome_net/input_batch_normalization/ReadVariableOp2n
5awsome_net/input_batch_normalization/ReadVariableOp_15awsome_net/input_batch_normalization/ReadVariableOp_12\
,awsome_net/input_conv/BiasAdd/ReadVariableOp,awsome_net/input_conv/BiasAdd/ReadVariableOp2Z
+awsome_net/input_conv/Conv2D/ReadVariableOp+awsome_net/input_conv/Conv2D/ReadVariableOp2z
;awsome_net/output_class_distribution/BiasAdd/ReadVariableOp;awsome_net/output_class_distribution/BiasAdd/ReadVariableOp2x
:awsome_net/output_class_distribution/MatMul/ReadVariableOp:awsome_net/output_class_distribution/MatMul/ReadVariableOp:0 ,
*
_user_specified_nameinput_conv_input
�
�
E__inference_b_norm_1_layer_call_and_return_conditional_losses_4353048

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity��FusedBatchNormV3/ReadVariableOp�!FusedBatchNormV3/ReadVariableOp_1�ReadVariableOp�ReadVariableOp_1^
LogicalAnd/xConst*
_output_shapes
: *
dtype0
*
value	B
 Z 2
LogicalAnd/x^
LogicalAnd/yConst*
_output_shapes
: *
dtype0
*
value	B
 Z2
LogicalAnd/yl

LogicalAnd
LogicalAndLogicalAnd/x:output:0LogicalAnd/y:output:0*
_output_shapes
: 2

LogicalAndt
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOp�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:���������@@:::::*
epsilon%o�:*
is_training( 2
FusedBatchNormV3S
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *�p}?2
Const�
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*/
_output_shapes
:���������@@2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:���������@@::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:& "
 
_user_specified_nameinputs
�
�
,__inference_input_conv_layer_call_fn_4352052

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*A
_output_shapes/
-:+���������������������������*-
config_proto

GPU

CPU2*0J 8*P
fKRI
G__inference_input_conv_layer_call_and_return_conditional_losses_43520442
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+���������������������������2

Identity"
identityIdentity:output:0*H
_input_shapes7
5:+���������������������������::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs
�$
�
V__inference_input_batch_normalization_layer_call_and_return_conditional_losses_4354308

inputs
readvariableop_resource
readvariableop_1_resource
assignmovingavg_4354293
assignmovingavg_1_4354300
identity��#AssignMovingAvg/AssignSubVariableOp�AssignMovingAvg/ReadVariableOp�%AssignMovingAvg_1/AssignSubVariableOp� AssignMovingAvg_1/ReadVariableOp�ReadVariableOp�ReadVariableOp_1^
LogicalAnd/xConst*
_output_shapes
: *
dtype0
*
value	B
 Z2
LogicalAnd/x^
LogicalAnd/yConst*
_output_shapes
: *
dtype0
*
value	B
 Z2
LogicalAnd/yl

LogicalAnd
LogicalAndLogicalAnd/x:output:0LogicalAnd/y:output:0*
_output_shapes
: 2

LogicalAndt
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1Q
ConstConst*
_output_shapes
: *
dtype0*
valueB 2
ConstU
Const_1Const*
_output_shapes
: *
dtype0*
valueB 2	
Const_1�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0Const:output:0Const_1:output:0*
T0*
U0*]
_output_shapesK
I:+���������������������������:::::*
epsilon%o�:2
FusedBatchNormV3W
Const_2Const*
_output_shapes
: *
dtype0*
valueB
 *�p}?2	
Const_2�
AssignMovingAvg/sub/xConst**
_class 
loc:@AssignMovingAvg/4354293*
_output_shapes
: *
dtype0*
valueB
 *  �?2
AssignMovingAvg/sub/x�
AssignMovingAvg/subSubAssignMovingAvg/sub/x:output:0Const_2:output:0*
T0**
_class 
loc:@AssignMovingAvg/4354293*
_output_shapes
: 2
AssignMovingAvg/sub�
AssignMovingAvg/ReadVariableOpReadVariableOpassignmovingavg_4354293*
_output_shapes
:*
dtype02 
AssignMovingAvg/ReadVariableOp�
AssignMovingAvg/sub_1Sub&AssignMovingAvg/ReadVariableOp:value:0FusedBatchNormV3:batch_mean:0*
T0**
_class 
loc:@AssignMovingAvg/4354293*
_output_shapes
:2
AssignMovingAvg/sub_1�
AssignMovingAvg/mulMulAssignMovingAvg/sub_1:z:0AssignMovingAvg/sub:z:0*
T0**
_class 
loc:@AssignMovingAvg/4354293*
_output_shapes
:2
AssignMovingAvg/mul�
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOpassignmovingavg_4354293AssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp**
_class 
loc:@AssignMovingAvg/4354293*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOp�
AssignMovingAvg_1/sub/xConst*,
_class"
 loc:@AssignMovingAvg_1/4354300*
_output_shapes
: *
dtype0*
valueB
 *  �?2
AssignMovingAvg_1/sub/x�
AssignMovingAvg_1/subSub AssignMovingAvg_1/sub/x:output:0Const_2:output:0*
T0*,
_class"
 loc:@AssignMovingAvg_1/4354300*
_output_shapes
: 2
AssignMovingAvg_1/sub�
 AssignMovingAvg_1/ReadVariableOpReadVariableOpassignmovingavg_1_4354300*
_output_shapes
:*
dtype02"
 AssignMovingAvg_1/ReadVariableOp�
AssignMovingAvg_1/sub_1Sub(AssignMovingAvg_1/ReadVariableOp:value:0!FusedBatchNormV3:batch_variance:0*
T0*,
_class"
 loc:@AssignMovingAvg_1/4354300*
_output_shapes
:2
AssignMovingAvg_1/sub_1�
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub_1:z:0AssignMovingAvg_1/sub:z:0*
T0*,
_class"
 loc:@AssignMovingAvg_1/4354300*
_output_shapes
:2
AssignMovingAvg_1/mul�
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOpassignmovingavg_1_4354300AssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*,
_class"
 loc:@AssignMovingAvg_1/4354300*
_output_shapes
 *
dtype02'
%AssignMovingAvg_1/AssignSubVariableOp�
IdentityIdentityFusedBatchNormV3:y:0$^AssignMovingAvg/AssignSubVariableOp^AssignMovingAvg/ReadVariableOp&^AssignMovingAvg_1/AssignSubVariableOp!^AssignMovingAvg_1/ReadVariableOp^ReadVariableOp^ReadVariableOp_1*
T0*A
_output_shapes/
-:+���������������������������2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+���������������������������::::2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp2D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp2 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:& "
 
_user_specified_nameinputs
�
�

,__inference_awsome_net_layer_call_fn_4353704
input_conv_input"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4"
statefulpartitionedcall_args_5"
statefulpartitionedcall_args_6"
statefulpartitionedcall_args_7"
statefulpartitionedcall_args_8"
statefulpartitionedcall_args_9#
statefulpartitionedcall_args_10#
statefulpartitionedcall_args_11#
statefulpartitionedcall_args_12#
statefulpartitionedcall_args_13#
statefulpartitionedcall_args_14#
statefulpartitionedcall_args_15#
statefulpartitionedcall_args_16#
statefulpartitionedcall_args_17#
statefulpartitionedcall_args_18#
statefulpartitionedcall_args_19#
statefulpartitionedcall_args_20#
statefulpartitionedcall_args_21#
statefulpartitionedcall_args_22#
statefulpartitionedcall_args_23#
statefulpartitionedcall_args_24#
statefulpartitionedcall_args_25#
statefulpartitionedcall_args_26#
statefulpartitionedcall_args_27#
statefulpartitionedcall_args_28#
statefulpartitionedcall_args_29#
statefulpartitionedcall_args_30#
statefulpartitionedcall_args_31#
statefulpartitionedcall_args_32
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinput_conv_inputstatefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4statefulpartitionedcall_args_5statefulpartitionedcall_args_6statefulpartitionedcall_args_7statefulpartitionedcall_args_8statefulpartitionedcall_args_9statefulpartitionedcall_args_10statefulpartitionedcall_args_11statefulpartitionedcall_args_12statefulpartitionedcall_args_13statefulpartitionedcall_args_14statefulpartitionedcall_args_15statefulpartitionedcall_args_16statefulpartitionedcall_args_17statefulpartitionedcall_args_18statefulpartitionedcall_args_19statefulpartitionedcall_args_20statefulpartitionedcall_args_21statefulpartitionedcall_args_22statefulpartitionedcall_args_23statefulpartitionedcall_args_24statefulpartitionedcall_args_25statefulpartitionedcall_args_26statefulpartitionedcall_args_27statefulpartitionedcall_args_28statefulpartitionedcall_args_29statefulpartitionedcall_args_30statefulpartitionedcall_args_31statefulpartitionedcall_args_32*,
Tin%
#2!*
Tout
2*,
_gradient_op_typePartitionedCallUnused*(
_output_shapes
:����������*-
config_proto

GPU

CPU2*0J 8*P
fKRI
G__inference_awsome_net_layer_call_and_return_conditional_losses_43536692
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:����������2

Identity"
identityIdentity:output:0*�
_input_shapes�
�:�����������::::::::::::::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:0 ,
*
_user_specified_nameinput_conv_input
�
i
M__inference_max_pooling2d_82_layer_call_and_return_conditional_losses_4352518

inputs
identity�
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4������������������������������������*
ksize
*
paddingVALID*
strides
2	
MaxPool�
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4������������������������������������2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4������������������������������������:& "
 
_user_specified_nameinputs
�
�

,__inference_awsome_net_layer_call_fn_4353607
input_conv_input"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4"
statefulpartitionedcall_args_5"
statefulpartitionedcall_args_6"
statefulpartitionedcall_args_7"
statefulpartitionedcall_args_8"
statefulpartitionedcall_args_9#
statefulpartitionedcall_args_10#
statefulpartitionedcall_args_11#
statefulpartitionedcall_args_12#
statefulpartitionedcall_args_13#
statefulpartitionedcall_args_14#
statefulpartitionedcall_args_15#
statefulpartitionedcall_args_16#
statefulpartitionedcall_args_17#
statefulpartitionedcall_args_18#
statefulpartitionedcall_args_19#
statefulpartitionedcall_args_20#
statefulpartitionedcall_args_21#
statefulpartitionedcall_args_22#
statefulpartitionedcall_args_23#
statefulpartitionedcall_args_24#
statefulpartitionedcall_args_25#
statefulpartitionedcall_args_26#
statefulpartitionedcall_args_27#
statefulpartitionedcall_args_28#
statefulpartitionedcall_args_29#
statefulpartitionedcall_args_30#
statefulpartitionedcall_args_31#
statefulpartitionedcall_args_32
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinput_conv_inputstatefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4statefulpartitionedcall_args_5statefulpartitionedcall_args_6statefulpartitionedcall_args_7statefulpartitionedcall_args_8statefulpartitionedcall_args_9statefulpartitionedcall_args_10statefulpartitionedcall_args_11statefulpartitionedcall_args_12statefulpartitionedcall_args_13statefulpartitionedcall_args_14statefulpartitionedcall_args_15statefulpartitionedcall_args_16statefulpartitionedcall_args_17statefulpartitionedcall_args_18statefulpartitionedcall_args_19statefulpartitionedcall_args_20statefulpartitionedcall_args_21statefulpartitionedcall_args_22statefulpartitionedcall_args_23statefulpartitionedcall_args_24statefulpartitionedcall_args_25statefulpartitionedcall_args_26statefulpartitionedcall_args_27statefulpartitionedcall_args_28statefulpartitionedcall_args_29statefulpartitionedcall_args_30statefulpartitionedcall_args_31statefulpartitionedcall_args_32*,
Tin%
#2!*
Tout
2*,
_gradient_op_typePartitionedCallUnused*(
_output_shapes
:����������*-
config_proto

GPU

CPU2*0J 8*P
fKRI
G__inference_awsome_net_layer_call_and_return_conditional_losses_43535722
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:����������2

Identity"
identityIdentity:output:0*�
_input_shapes�
�:�����������::::::::::::::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:0 ,
*
_user_specified_nameinput_conv_input
�
�
E__inference_b_norm_2_layer_call_and_return_conditional_losses_4354705

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity��FusedBatchNormV3/ReadVariableOp�!FusedBatchNormV3/ReadVariableOp_1�ReadVariableOp�ReadVariableOp_1^
LogicalAnd/xConst*
_output_shapes
: *
dtype0
*
value	B
 Z 2
LogicalAnd/x^
LogicalAnd/yConst*
_output_shapes
: *
dtype0
*
value	B
 Z2
LogicalAnd/yl

LogicalAnd
LogicalAndLogicalAnd/x:output:0LogicalAnd/y:output:0*
_output_shapes
: 2

LogicalAndt
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype02
ReadVariableOp_1�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02!
FusedBatchNormV3/ReadVariableOp�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02#
!FusedBatchNormV3/ReadVariableOp_1�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+��������������������������� : : : : :*
epsilon%o�:*
is_training( 2
FusedBatchNormV3S
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *�p}?2
Const�
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*A
_output_shapes/
-:+��������������������������� 2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+��������������������������� ::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:& "
 
_user_specified_nameinputs
�
�
E__inference_b_norm_4_layer_call_and_return_conditional_losses_4355119

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity��FusedBatchNormV3/ReadVariableOp�!FusedBatchNormV3/ReadVariableOp_1�ReadVariableOp�ReadVariableOp_1^
LogicalAnd/xConst*
_output_shapes
: *
dtype0
*
value	B
 Z 2
LogicalAnd/x^
LogicalAnd/yConst*
_output_shapes
: *
dtype0
*
value	B
 Z2
LogicalAnd/yl

LogicalAnd
LogicalAndLogicalAnd/x:output:0LogicalAnd/y:output:0*
_output_shapes
: 2

LogicalAndt
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOp�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:���������@:@:@:@:@:*
epsilon%o�:*
is_training( 2
FusedBatchNormV3S
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *�p}?2
Const�
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*/
_output_shapes
:���������@2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:���������@::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:& "
 
_user_specified_nameinputs
�
�
E__inference_b_norm_3_layer_call_and_return_conditional_losses_4354875

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity��FusedBatchNormV3/ReadVariableOp�!FusedBatchNormV3/ReadVariableOp_1�ReadVariableOp�ReadVariableOp_1^
LogicalAnd/xConst*
_output_shapes
: *
dtype0
*
value	B
 Z 2
LogicalAnd/x^
LogicalAnd/yConst*
_output_shapes
: *
dtype0
*
value	B
 Z2
LogicalAnd/yl

LogicalAnd
LogicalAndLogicalAnd/x:output:0LogicalAnd/y:output:0*
_output_shapes
: 2

LogicalAndt
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOp�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+���������������������������@:@:@:@:@:*
epsilon%o�:*
is_training( 2
FusedBatchNormV3S
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *�p}?2
Const�
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*A
_output_shapes/
-:+���������������������������@2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+���������������������������@::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:& "
 
_user_specified_nameinputs
�
�
E__inference_b_norm_2_layer_call_and_return_conditional_losses_4354779

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity��FusedBatchNormV3/ReadVariableOp�!FusedBatchNormV3/ReadVariableOp_1�ReadVariableOp�ReadVariableOp_1^
LogicalAnd/xConst*
_output_shapes
: *
dtype0
*
value	B
 Z 2
LogicalAnd/x^
LogicalAnd/yConst*
_output_shapes
: *
dtype0
*
value	B
 Z2
LogicalAnd/yl

LogicalAnd
LogicalAndLogicalAnd/x:output:0LogicalAnd/y:output:0*
_output_shapes
: 2

LogicalAndt
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype02
ReadVariableOp_1�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02!
FusedBatchNormV3/ReadVariableOp�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02#
!FusedBatchNormV3/ReadVariableOp_1�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:���������   : : : : :*
epsilon%o�:*
is_training( 2
FusedBatchNormV3S
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *�p}?2
Const�
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*/
_output_shapes
:���������   2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:���������   ::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:& "
 
_user_specified_nameinputs
�$
�
E__inference_b_norm_2_layer_call_and_return_conditional_losses_4353122

inputs
readvariableop_resource
readvariableop_1_resource
assignmovingavg_4353107
assignmovingavg_1_4353114
identity��#AssignMovingAvg/AssignSubVariableOp�AssignMovingAvg/ReadVariableOp�%AssignMovingAvg_1/AssignSubVariableOp� AssignMovingAvg_1/ReadVariableOp�ReadVariableOp�ReadVariableOp_1^
LogicalAnd/xConst*
_output_shapes
: *
dtype0
*
value	B
 Z2
LogicalAnd/x^
LogicalAnd/yConst*
_output_shapes
: *
dtype0
*
value	B
 Z2
LogicalAnd/yl

LogicalAnd
LogicalAndLogicalAnd/x:output:0LogicalAnd/y:output:0*
_output_shapes
: 2

LogicalAndt
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype02
ReadVariableOp_1Q
ConstConst*
_output_shapes
: *
dtype0*
valueB 2
ConstU
Const_1Const*
_output_shapes
: *
dtype0*
valueB 2	
Const_1�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0Const:output:0Const_1:output:0*
T0*
U0*K
_output_shapes9
7:���������   : : : : :*
epsilon%o�:2
FusedBatchNormV3W
Const_2Const*
_output_shapes
: *
dtype0*
valueB
 *�p}?2	
Const_2�
AssignMovingAvg/sub/xConst**
_class 
loc:@AssignMovingAvg/4353107*
_output_shapes
: *
dtype0*
valueB
 *  �?2
AssignMovingAvg/sub/x�
AssignMovingAvg/subSubAssignMovingAvg/sub/x:output:0Const_2:output:0*
T0**
_class 
loc:@AssignMovingAvg/4353107*
_output_shapes
: 2
AssignMovingAvg/sub�
AssignMovingAvg/ReadVariableOpReadVariableOpassignmovingavg_4353107*
_output_shapes
: *
dtype02 
AssignMovingAvg/ReadVariableOp�
AssignMovingAvg/sub_1Sub&AssignMovingAvg/ReadVariableOp:value:0FusedBatchNormV3:batch_mean:0*
T0**
_class 
loc:@AssignMovingAvg/4353107*
_output_shapes
: 2
AssignMovingAvg/sub_1�
AssignMovingAvg/mulMulAssignMovingAvg/sub_1:z:0AssignMovingAvg/sub:z:0*
T0**
_class 
loc:@AssignMovingAvg/4353107*
_output_shapes
: 2
AssignMovingAvg/mul�
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOpassignmovingavg_4353107AssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp**
_class 
loc:@AssignMovingAvg/4353107*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOp�
AssignMovingAvg_1/sub/xConst*,
_class"
 loc:@AssignMovingAvg_1/4353114*
_output_shapes
: *
dtype0*
valueB
 *  �?2
AssignMovingAvg_1/sub/x�
AssignMovingAvg_1/subSub AssignMovingAvg_1/sub/x:output:0Const_2:output:0*
T0*,
_class"
 loc:@AssignMovingAvg_1/4353114*
_output_shapes
: 2
AssignMovingAvg_1/sub�
 AssignMovingAvg_1/ReadVariableOpReadVariableOpassignmovingavg_1_4353114*
_output_shapes
: *
dtype02"
 AssignMovingAvg_1/ReadVariableOp�
AssignMovingAvg_1/sub_1Sub(AssignMovingAvg_1/ReadVariableOp:value:0!FusedBatchNormV3:batch_variance:0*
T0*,
_class"
 loc:@AssignMovingAvg_1/4353114*
_output_shapes
: 2
AssignMovingAvg_1/sub_1�
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub_1:z:0AssignMovingAvg_1/sub:z:0*
T0*,
_class"
 loc:@AssignMovingAvg_1/4353114*
_output_shapes
: 2
AssignMovingAvg_1/mul�
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOpassignmovingavg_1_4353114AssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*,
_class"
 loc:@AssignMovingAvg_1/4353114*
_output_shapes
 *
dtype02'
%AssignMovingAvg_1/AssignSubVariableOp�
IdentityIdentityFusedBatchNormV3:y:0$^AssignMovingAvg/AssignSubVariableOp^AssignMovingAvg/ReadVariableOp&^AssignMovingAvg_1/AssignSubVariableOp!^AssignMovingAvg_1/ReadVariableOp^ReadVariableOp^ReadVariableOp_1*
T0*/
_output_shapes
:���������   2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:���������   ::::2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp2D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp2 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:& "
 
_user_specified_nameinputs
�p
�
G__inference_awsome_net_layer_call_and_return_conditional_losses_4353509
input_conv_input-
)input_conv_statefulpartitionedcall_args_1-
)input_conv_statefulpartitionedcall_args_2<
8input_batch_normalization_statefulpartitionedcall_args_1<
8input_batch_normalization_statefulpartitionedcall_args_2<
8input_batch_normalization_statefulpartitionedcall_args_3<
8input_batch_normalization_statefulpartitionedcall_args_4)
%conv_1_statefulpartitionedcall_args_1)
%conv_1_statefulpartitionedcall_args_2+
'b_norm_1_statefulpartitionedcall_args_1+
'b_norm_1_statefulpartitionedcall_args_2+
'b_norm_1_statefulpartitionedcall_args_3+
'b_norm_1_statefulpartitionedcall_args_4)
%conv_2_statefulpartitionedcall_args_1)
%conv_2_statefulpartitionedcall_args_2+
'b_norm_2_statefulpartitionedcall_args_1+
'b_norm_2_statefulpartitionedcall_args_2+
'b_norm_2_statefulpartitionedcall_args_3+
'b_norm_2_statefulpartitionedcall_args_4)
%conv_3_statefulpartitionedcall_args_1)
%conv_3_statefulpartitionedcall_args_2+
'b_norm_3_statefulpartitionedcall_args_1+
'b_norm_3_statefulpartitionedcall_args_2+
'b_norm_3_statefulpartitionedcall_args_3+
'b_norm_3_statefulpartitionedcall_args_4)
%conv_4_statefulpartitionedcall_args_1)
%conv_4_statefulpartitionedcall_args_2+
'b_norm_4_statefulpartitionedcall_args_1+
'b_norm_4_statefulpartitionedcall_args_2+
'b_norm_4_statefulpartitionedcall_args_3+
'b_norm_4_statefulpartitionedcall_args_4<
8output_class_distribution_statefulpartitionedcall_args_1<
8output_class_distribution_statefulpartitionedcall_args_2
identity�� b_norm_1/StatefulPartitionedCall� b_norm_2/StatefulPartitionedCall� b_norm_3/StatefulPartitionedCall� b_norm_4/StatefulPartitionedCall�conv_1/StatefulPartitionedCall�conv_2/StatefulPartitionedCall�conv_3/StatefulPartitionedCall�conv_4/StatefulPartitionedCall�1input_batch_normalization/StatefulPartitionedCall�"input_conv/StatefulPartitionedCall�1output_class_distribution/StatefulPartitionedCall�
"input_conv/StatefulPartitionedCallStatefulPartitionedCallinput_conv_input)input_conv_statefulpartitionedcall_args_1)input_conv_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*1
_output_shapes
:�����������*-
config_proto

GPU

CPU2*0J 8*P
fKRI
G__inference_input_conv_layer_call_and_return_conditional_losses_43520442$
"input_conv/StatefulPartitionedCall�
1input_batch_normalization/StatefulPartitionedCallStatefulPartitionedCall+input_conv/StatefulPartitionedCall:output:08input_batch_normalization_statefulpartitionedcall_args_18input_batch_normalization_statefulpartitionedcall_args_28input_batch_normalization_statefulpartitionedcall_args_38input_batch_normalization_statefulpartitionedcall_args_4*
Tin	
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*1
_output_shapes
:�����������*-
config_proto

GPU

CPU2*0J 8*_
fZRX
V__inference_input_batch_normalization_layer_call_and_return_conditional_losses_435291423
1input_batch_normalization/StatefulPartitionedCall�
input_RELU/PartitionedCallPartitionedCall:input_batch_normalization/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*1
_output_shapes
:�����������*-
config_proto

GPU

CPU2*0J 8*P
fKRI
G__inference_input_RELU_layer_call_and_return_conditional_losses_43529432
input_RELU/PartitionedCall�
 max_pooling2d_85/PartitionedCallPartitionedCall#input_RELU/PartitionedCall:output:0*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*/
_output_shapes
:���������@@*-
config_proto

GPU

CPU2*0J 8*V
fQRO
M__inference_max_pooling2d_85_layer_call_and_return_conditional_losses_43521902"
 max_pooling2d_85/PartitionedCall�
dropout_42/PartitionedCallPartitionedCall)max_pooling2d_85/PartitionedCall:output:0*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*/
_output_shapes
:���������@@*-
config_proto

GPU

CPU2*0J 8*P
fKRI
G__inference_dropout_42_layer_call_and_return_conditional_losses_43529772
dropout_42/PartitionedCall�
conv_1/StatefulPartitionedCallStatefulPartitionedCall#dropout_42/PartitionedCall:output:0%conv_1_statefulpartitionedcall_args_1%conv_1_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*/
_output_shapes
:���������@@*-
config_proto

GPU

CPU2*0J 8*L
fGRE
C__inference_conv_1_layer_call_and_return_conditional_losses_43522082 
conv_1/StatefulPartitionedCall�
 b_norm_1/StatefulPartitionedCallStatefulPartitionedCall'conv_1/StatefulPartitionedCall:output:0'b_norm_1_statefulpartitionedcall_args_1'b_norm_1_statefulpartitionedcall_args_2'b_norm_1_statefulpartitionedcall_args_3'b_norm_1_statefulpartitionedcall_args_4*
Tin	
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*/
_output_shapes
:���������@@*-
config_proto

GPU

CPU2*0J 8*N
fIRG
E__inference_b_norm_1_layer_call_and_return_conditional_losses_43530482"
 b_norm_1/StatefulPartitionedCall�
RELU_1/PartitionedCallPartitionedCall)b_norm_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*/
_output_shapes
:���������@@*-
config_proto

GPU

CPU2*0J 8*L
fGRE
C__inference_RELU_1_layer_call_and_return_conditional_losses_43530772
RELU_1/PartitionedCall�
 max_pooling2d_81/PartitionedCallPartitionedCallRELU_1/PartitionedCall:output:0*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*/
_output_shapes
:���������  *-
config_proto

GPU

CPU2*0J 8*V
fQRO
M__inference_max_pooling2d_81_layer_call_and_return_conditional_losses_43523542"
 max_pooling2d_81/PartitionedCall�
conv_2/StatefulPartitionedCallStatefulPartitionedCall)max_pooling2d_81/PartitionedCall:output:0%conv_2_statefulpartitionedcall_args_1%conv_2_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*/
_output_shapes
:���������   *-
config_proto

GPU

CPU2*0J 8*L
fGRE
C__inference_conv_2_layer_call_and_return_conditional_losses_43523722 
conv_2/StatefulPartitionedCall�
 b_norm_2/StatefulPartitionedCallStatefulPartitionedCall'conv_2/StatefulPartitionedCall:output:0'b_norm_2_statefulpartitionedcall_args_1'b_norm_2_statefulpartitionedcall_args_2'b_norm_2_statefulpartitionedcall_args_3'b_norm_2_statefulpartitionedcall_args_4*
Tin	
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*/
_output_shapes
:���������   *-
config_proto

GPU

CPU2*0J 8*N
fIRG
E__inference_b_norm_2_layer_call_and_return_conditional_losses_43531442"
 b_norm_2/StatefulPartitionedCall�
RELU_2/PartitionedCallPartitionedCall)b_norm_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*/
_output_shapes
:���������   *-
config_proto

GPU

CPU2*0J 8*L
fGRE
C__inference_RELU_2_layer_call_and_return_conditional_losses_43531732
RELU_2/PartitionedCall�
 max_pooling2d_82/PartitionedCallPartitionedCallRELU_2/PartitionedCall:output:0*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*/
_output_shapes
:��������� *-
config_proto

GPU

CPU2*0J 8*V
fQRO
M__inference_max_pooling2d_82_layer_call_and_return_conditional_losses_43525182"
 max_pooling2d_82/PartitionedCall�
conv_3/StatefulPartitionedCallStatefulPartitionedCall)max_pooling2d_82/PartitionedCall:output:0%conv_3_statefulpartitionedcall_args_1%conv_3_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*/
_output_shapes
:���������@*-
config_proto

GPU

CPU2*0J 8*L
fGRE
C__inference_conv_3_layer_call_and_return_conditional_losses_43525362 
conv_3/StatefulPartitionedCall�
 b_norm_3/StatefulPartitionedCallStatefulPartitionedCall'conv_3/StatefulPartitionedCall:output:0'b_norm_3_statefulpartitionedcall_args_1'b_norm_3_statefulpartitionedcall_args_2'b_norm_3_statefulpartitionedcall_args_3'b_norm_3_statefulpartitionedcall_args_4*
Tin	
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*/
_output_shapes
:���������@*-
config_proto

GPU

CPU2*0J 8*N
fIRG
E__inference_b_norm_3_layer_call_and_return_conditional_losses_43532402"
 b_norm_3/StatefulPartitionedCall�
RELU_3/PartitionedCallPartitionedCall)b_norm_3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*/
_output_shapes
:���������@*-
config_proto

GPU

CPU2*0J 8*L
fGRE
C__inference_RELU_3_layer_call_and_return_conditional_losses_43532692
RELU_3/PartitionedCall�
 max_pooling2d_83/PartitionedCallPartitionedCallRELU_3/PartitionedCall:output:0*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*/
_output_shapes
:���������@*-
config_proto

GPU

CPU2*0J 8*V
fQRO
M__inference_max_pooling2d_83_layer_call_and_return_conditional_losses_43526822"
 max_pooling2d_83/PartitionedCall�
conv_4/StatefulPartitionedCallStatefulPartitionedCall)max_pooling2d_83/PartitionedCall:output:0%conv_4_statefulpartitionedcall_args_1%conv_4_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*/
_output_shapes
:���������@*-
config_proto

GPU

CPU2*0J 8*L
fGRE
C__inference_conv_4_layer_call_and_return_conditional_losses_43527002 
conv_4/StatefulPartitionedCall�
 b_norm_4/StatefulPartitionedCallStatefulPartitionedCall'conv_4/StatefulPartitionedCall:output:0'b_norm_4_statefulpartitionedcall_args_1'b_norm_4_statefulpartitionedcall_args_2'b_norm_4_statefulpartitionedcall_args_3'b_norm_4_statefulpartitionedcall_args_4*
Tin	
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*/
_output_shapes
:���������@*-
config_proto

GPU

CPU2*0J 8*N
fIRG
E__inference_b_norm_4_layer_call_and_return_conditional_losses_43533362"
 b_norm_4/StatefulPartitionedCall�
RELU_4/PartitionedCallPartitionedCall)b_norm_4/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*/
_output_shapes
:���������@*-
config_proto

GPU

CPU2*0J 8*L
fGRE
C__inference_RELU_4_layer_call_and_return_conditional_losses_43533652
RELU_4/PartitionedCall�
 max_pooling2d_84/PartitionedCallPartitionedCallRELU_4/PartitionedCall:output:0*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*/
_output_shapes
:���������@*-
config_proto

GPU

CPU2*0J 8*V
fQRO
M__inference_max_pooling2d_84_layer_call_and_return_conditional_losses_43528462"
 max_pooling2d_84/PartitionedCall�
flatten_21/PartitionedCallPartitionedCall)max_pooling2d_84/PartitionedCall:output:0*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*(
_output_shapes
:����������*-
config_proto

GPU

CPU2*0J 8*P
fKRI
G__inference_flatten_21_layer_call_and_return_conditional_losses_43533802
flatten_21/PartitionedCall�
dropout_43/PartitionedCallPartitionedCall#flatten_21/PartitionedCall:output:0*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*(
_output_shapes
:����������*-
config_proto

GPU

CPU2*0J 8*P
fKRI
G__inference_dropout_43_layer_call_and_return_conditional_losses_43534132
dropout_43/PartitionedCall�
1output_class_distribution/StatefulPartitionedCallStatefulPartitionedCall#dropout_43/PartitionedCall:output:08output_class_distribution_statefulpartitionedcall_args_18output_class_distribution_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*(
_output_shapes
:����������*-
config_proto

GPU

CPU2*0J 8*_
fZRX
V__inference_output_class_distribution_layer_call_and_return_conditional_losses_435343623
1output_class_distribution/StatefulPartitionedCall�
IdentityIdentity:output_class_distribution/StatefulPartitionedCall:output:0!^b_norm_1/StatefulPartitionedCall!^b_norm_2/StatefulPartitionedCall!^b_norm_3/StatefulPartitionedCall!^b_norm_4/StatefulPartitionedCall^conv_1/StatefulPartitionedCall^conv_2/StatefulPartitionedCall^conv_3/StatefulPartitionedCall^conv_4/StatefulPartitionedCall2^input_batch_normalization/StatefulPartitionedCall#^input_conv/StatefulPartitionedCall2^output_class_distribution/StatefulPartitionedCall*
T0*(
_output_shapes
:����������2

Identity"
identityIdentity:output:0*�
_input_shapes�
�:�����������::::::::::::::::::::::::::::::::2D
 b_norm_1/StatefulPartitionedCall b_norm_1/StatefulPartitionedCall2D
 b_norm_2/StatefulPartitionedCall b_norm_2/StatefulPartitionedCall2D
 b_norm_3/StatefulPartitionedCall b_norm_3/StatefulPartitionedCall2D
 b_norm_4/StatefulPartitionedCall b_norm_4/StatefulPartitionedCall2@
conv_1/StatefulPartitionedCallconv_1/StatefulPartitionedCall2@
conv_2/StatefulPartitionedCallconv_2/StatefulPartitionedCall2@
conv_3/StatefulPartitionedCallconv_3/StatefulPartitionedCall2@
conv_4/StatefulPartitionedCallconv_4/StatefulPartitionedCall2f
1input_batch_normalization/StatefulPartitionedCall1input_batch_normalization/StatefulPartitionedCall2H
"input_conv/StatefulPartitionedCall"input_conv/StatefulPartitionedCall2f
1output_class_distribution/StatefulPartitionedCall1output_class_distribution/StatefulPartitionedCall:0 ,
*
_user_specified_nameinput_conv_input
�
�
E__inference_b_norm_3_layer_call_and_return_conditional_losses_4354949

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity��FusedBatchNormV3/ReadVariableOp�!FusedBatchNormV3/ReadVariableOp_1�ReadVariableOp�ReadVariableOp_1^
LogicalAnd/xConst*
_output_shapes
: *
dtype0
*
value	B
 Z 2
LogicalAnd/x^
LogicalAnd/yConst*
_output_shapes
: *
dtype0
*
value	B
 Z2
LogicalAnd/yl

LogicalAnd
LogicalAndLogicalAnd/x:output:0LogicalAnd/y:output:0*
_output_shapes
: 2

LogicalAndt
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOp�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:���������@:@:@:@:@:*
epsilon%o�:*
is_training( 2
FusedBatchNormV3S
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *�p}?2
Const�
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*/
_output_shapes
:���������@2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:���������@::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:& "
 
_user_specified_nameinputs
�$
�
E__inference_b_norm_4_layer_call_and_return_conditional_losses_4355023

inputs
readvariableop_resource
readvariableop_1_resource
assignmovingavg_4355008
assignmovingavg_1_4355015
identity��#AssignMovingAvg/AssignSubVariableOp�AssignMovingAvg/ReadVariableOp�%AssignMovingAvg_1/AssignSubVariableOp� AssignMovingAvg_1/ReadVariableOp�ReadVariableOp�ReadVariableOp_1^
LogicalAnd/xConst*
_output_shapes
: *
dtype0
*
value	B
 Z2
LogicalAnd/x^
LogicalAnd/yConst*
_output_shapes
: *
dtype0
*
value	B
 Z2
LogicalAnd/yl

LogicalAnd
LogicalAndLogicalAnd/x:output:0LogicalAnd/y:output:0*
_output_shapes
: 2

LogicalAndt
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1Q
ConstConst*
_output_shapes
: *
dtype0*
valueB 2
ConstU
Const_1Const*
_output_shapes
: *
dtype0*
valueB 2	
Const_1�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0Const:output:0Const_1:output:0*
T0*
U0*]
_output_shapesK
I:+���������������������������@:@:@:@:@:*
epsilon%o�:2
FusedBatchNormV3W
Const_2Const*
_output_shapes
: *
dtype0*
valueB
 *�p}?2	
Const_2�
AssignMovingAvg/sub/xConst**
_class 
loc:@AssignMovingAvg/4355008*
_output_shapes
: *
dtype0*
valueB
 *  �?2
AssignMovingAvg/sub/x�
AssignMovingAvg/subSubAssignMovingAvg/sub/x:output:0Const_2:output:0*
T0**
_class 
loc:@AssignMovingAvg/4355008*
_output_shapes
: 2
AssignMovingAvg/sub�
AssignMovingAvg/ReadVariableOpReadVariableOpassignmovingavg_4355008*
_output_shapes
:@*
dtype02 
AssignMovingAvg/ReadVariableOp�
AssignMovingAvg/sub_1Sub&AssignMovingAvg/ReadVariableOp:value:0FusedBatchNormV3:batch_mean:0*
T0**
_class 
loc:@AssignMovingAvg/4355008*
_output_shapes
:@2
AssignMovingAvg/sub_1�
AssignMovingAvg/mulMulAssignMovingAvg/sub_1:z:0AssignMovingAvg/sub:z:0*
T0**
_class 
loc:@AssignMovingAvg/4355008*
_output_shapes
:@2
AssignMovingAvg/mul�
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOpassignmovingavg_4355008AssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp**
_class 
loc:@AssignMovingAvg/4355008*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOp�
AssignMovingAvg_1/sub/xConst*,
_class"
 loc:@AssignMovingAvg_1/4355015*
_output_shapes
: *
dtype0*
valueB
 *  �?2
AssignMovingAvg_1/sub/x�
AssignMovingAvg_1/subSub AssignMovingAvg_1/sub/x:output:0Const_2:output:0*
T0*,
_class"
 loc:@AssignMovingAvg_1/4355015*
_output_shapes
: 2
AssignMovingAvg_1/sub�
 AssignMovingAvg_1/ReadVariableOpReadVariableOpassignmovingavg_1_4355015*
_output_shapes
:@*
dtype02"
 AssignMovingAvg_1/ReadVariableOp�
AssignMovingAvg_1/sub_1Sub(AssignMovingAvg_1/ReadVariableOp:value:0!FusedBatchNormV3:batch_variance:0*
T0*,
_class"
 loc:@AssignMovingAvg_1/4355015*
_output_shapes
:@2
AssignMovingAvg_1/sub_1�
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub_1:z:0AssignMovingAvg_1/sub:z:0*
T0*,
_class"
 loc:@AssignMovingAvg_1/4355015*
_output_shapes
:@2
AssignMovingAvg_1/mul�
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOpassignmovingavg_1_4355015AssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*,
_class"
 loc:@AssignMovingAvg_1/4355015*
_output_shapes
 *
dtype02'
%AssignMovingAvg_1/AssignSubVariableOp�
IdentityIdentityFusedBatchNormV3:y:0$^AssignMovingAvg/AssignSubVariableOp^AssignMovingAvg/ReadVariableOp&^AssignMovingAvg_1/AssignSubVariableOp!^AssignMovingAvg_1/ReadVariableOp^ReadVariableOp^ReadVariableOp_1*
T0*A
_output_shapes/
-:+���������������������������@2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+���������������������������@::::2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp2D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp2 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:& "
 
_user_specified_nameinputs
�
_
C__inference_RELU_3_layer_call_and_return_conditional_losses_4354972

inputs
identityV
ReluReluinputs*
T0*/
_output_shapes
:���������@2
Relun
IdentityIdentityRelu:activations:0*
T0*/
_output_shapes
:���������@2

Identity"
identityIdentity:output:0*.
_input_shapes
:���������@:& "
 
_user_specified_nameinputs
�

�
C__inference_conv_3_layer_call_and_return_conditional_losses_4352536

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOpo
dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      2
dilation_rate�
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: @*
dtype02
Conv2D/ReadVariableOp�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+���������������������������@*
paddingSAME*
strides
2
Conv2D�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+���������������������������@2	
BiasAdd�
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*A
_output_shapes/
-:+���������������������������@2

Identity"
identityIdentity:output:0*H
_input_shapes7
5:+��������������������������� ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:& "
 
_user_specified_nameinputs
�
�
E__inference_b_norm_4_layer_call_and_return_conditional_losses_4355045

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity��FusedBatchNormV3/ReadVariableOp�!FusedBatchNormV3/ReadVariableOp_1�ReadVariableOp�ReadVariableOp_1^
LogicalAnd/xConst*
_output_shapes
: *
dtype0
*
value	B
 Z 2
LogicalAnd/x^
LogicalAnd/yConst*
_output_shapes
: *
dtype0
*
value	B
 Z2
LogicalAnd/yl

LogicalAnd
LogicalAndLogicalAnd/x:output:0LogicalAnd/y:output:0*
_output_shapes
: 2

LogicalAndt
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOp�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+���������������������������@:@:@:@:@:*
epsilon%o�:*
is_training( 2
FusedBatchNormV3S
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *�p}?2
Const�
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*A
_output_shapes/
-:+���������������������������@2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+���������������������������@::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:& "
 
_user_specified_nameinputs
�
�
(__inference_conv_1_layer_call_fn_4352216

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*A
_output_shapes/
-:+���������������������������*-
config_proto

GPU

CPU2*0J 8*L
fGRE
C__inference_conv_1_layer_call_and_return_conditional_losses_43522082
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+���������������������������2

Identity"
identityIdentity:output:0*H
_input_shapes7
5:+���������������������������::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs
�$
�
E__inference_b_norm_3_layer_call_and_return_conditional_losses_4352638

inputs
readvariableop_resource
readvariableop_1_resource
assignmovingavg_4352623
assignmovingavg_1_4352630
identity��#AssignMovingAvg/AssignSubVariableOp�AssignMovingAvg/ReadVariableOp�%AssignMovingAvg_1/AssignSubVariableOp� AssignMovingAvg_1/ReadVariableOp�ReadVariableOp�ReadVariableOp_1^
LogicalAnd/xConst*
_output_shapes
: *
dtype0
*
value	B
 Z2
LogicalAnd/x^
LogicalAnd/yConst*
_output_shapes
: *
dtype0
*
value	B
 Z2
LogicalAnd/yl

LogicalAnd
LogicalAndLogicalAnd/x:output:0LogicalAnd/y:output:0*
_output_shapes
: 2

LogicalAndt
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1Q
ConstConst*
_output_shapes
: *
dtype0*
valueB 2
ConstU
Const_1Const*
_output_shapes
: *
dtype0*
valueB 2	
Const_1�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0Const:output:0Const_1:output:0*
T0*
U0*]
_output_shapesK
I:+���������������������������@:@:@:@:@:*
epsilon%o�:2
FusedBatchNormV3W
Const_2Const*
_output_shapes
: *
dtype0*
valueB
 *�p}?2	
Const_2�
AssignMovingAvg/sub/xConst**
_class 
loc:@AssignMovingAvg/4352623*
_output_shapes
: *
dtype0*
valueB
 *  �?2
AssignMovingAvg/sub/x�
AssignMovingAvg/subSubAssignMovingAvg/sub/x:output:0Const_2:output:0*
T0**
_class 
loc:@AssignMovingAvg/4352623*
_output_shapes
: 2
AssignMovingAvg/sub�
AssignMovingAvg/ReadVariableOpReadVariableOpassignmovingavg_4352623*
_output_shapes
:@*
dtype02 
AssignMovingAvg/ReadVariableOp�
AssignMovingAvg/sub_1Sub&AssignMovingAvg/ReadVariableOp:value:0FusedBatchNormV3:batch_mean:0*
T0**
_class 
loc:@AssignMovingAvg/4352623*
_output_shapes
:@2
AssignMovingAvg/sub_1�
AssignMovingAvg/mulMulAssignMovingAvg/sub_1:z:0AssignMovingAvg/sub:z:0*
T0**
_class 
loc:@AssignMovingAvg/4352623*
_output_shapes
:@2
AssignMovingAvg/mul�
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOpassignmovingavg_4352623AssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp**
_class 
loc:@AssignMovingAvg/4352623*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOp�
AssignMovingAvg_1/sub/xConst*,
_class"
 loc:@AssignMovingAvg_1/4352630*
_output_shapes
: *
dtype0*
valueB
 *  �?2
AssignMovingAvg_1/sub/x�
AssignMovingAvg_1/subSub AssignMovingAvg_1/sub/x:output:0Const_2:output:0*
T0*,
_class"
 loc:@AssignMovingAvg_1/4352630*
_output_shapes
: 2
AssignMovingAvg_1/sub�
 AssignMovingAvg_1/ReadVariableOpReadVariableOpassignmovingavg_1_4352630*
_output_shapes
:@*
dtype02"
 AssignMovingAvg_1/ReadVariableOp�
AssignMovingAvg_1/sub_1Sub(AssignMovingAvg_1/ReadVariableOp:value:0!FusedBatchNormV3:batch_variance:0*
T0*,
_class"
 loc:@AssignMovingAvg_1/4352630*
_output_shapes
:@2
AssignMovingAvg_1/sub_1�
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub_1:z:0AssignMovingAvg_1/sub:z:0*
T0*,
_class"
 loc:@AssignMovingAvg_1/4352630*
_output_shapes
:@2
AssignMovingAvg_1/mul�
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOpassignmovingavg_1_4352630AssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*,
_class"
 loc:@AssignMovingAvg_1/4352630*
_output_shapes
 *
dtype02'
%AssignMovingAvg_1/AssignSubVariableOp�
IdentityIdentityFusedBatchNormV3:y:0$^AssignMovingAvg/AssignSubVariableOp^AssignMovingAvg/ReadVariableOp&^AssignMovingAvg_1/AssignSubVariableOp!^AssignMovingAvg_1/ReadVariableOp^ReadVariableOp^ReadVariableOp_1*
T0*A
_output_shapes/
-:+���������������������������@2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+���������������������������@::::2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp2D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp2 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:& "
 
_user_specified_nameinputs
��
�
G__inference_awsome_net_layer_call_and_return_conditional_losses_4354044

inputs-
)input_conv_conv2d_readvariableop_resource.
*input_conv_biasadd_readvariableop_resource5
1input_batch_normalization_readvariableop_resource7
3input_batch_normalization_readvariableop_1_resource5
1input_batch_normalization_assignmovingavg_43538357
3input_batch_normalization_assignmovingavg_1_4353842)
%conv_1_conv2d_readvariableop_resource*
&conv_1_biasadd_readvariableop_resource$
 b_norm_1_readvariableop_resource&
"b_norm_1_readvariableop_1_resource$
 b_norm_1_assignmovingavg_4353889&
"b_norm_1_assignmovingavg_1_4353896)
%conv_2_conv2d_readvariableop_resource*
&conv_2_biasadd_readvariableop_resource$
 b_norm_2_readvariableop_resource&
"b_norm_2_readvariableop_1_resource$
 b_norm_2_assignmovingavg_4353927&
"b_norm_2_assignmovingavg_1_4353934)
%conv_3_conv2d_readvariableop_resource*
&conv_3_biasadd_readvariableop_resource$
 b_norm_3_readvariableop_resource&
"b_norm_3_readvariableop_1_resource$
 b_norm_3_assignmovingavg_4353965&
"b_norm_3_assignmovingavg_1_4353972)
%conv_4_conv2d_readvariableop_resource*
&conv_4_biasadd_readvariableop_resource$
 b_norm_4_readvariableop_resource&
"b_norm_4_readvariableop_1_resource$
 b_norm_4_assignmovingavg_4354003&
"b_norm_4_assignmovingavg_1_4354010<
8output_class_distribution_matmul_readvariableop_resource=
9output_class_distribution_biasadd_readvariableop_resource
identity��,b_norm_1/AssignMovingAvg/AssignSubVariableOp�'b_norm_1/AssignMovingAvg/ReadVariableOp�.b_norm_1/AssignMovingAvg_1/AssignSubVariableOp�)b_norm_1/AssignMovingAvg_1/ReadVariableOp�b_norm_1/ReadVariableOp�b_norm_1/ReadVariableOp_1�,b_norm_2/AssignMovingAvg/AssignSubVariableOp�'b_norm_2/AssignMovingAvg/ReadVariableOp�.b_norm_2/AssignMovingAvg_1/AssignSubVariableOp�)b_norm_2/AssignMovingAvg_1/ReadVariableOp�b_norm_2/ReadVariableOp�b_norm_2/ReadVariableOp_1�,b_norm_3/AssignMovingAvg/AssignSubVariableOp�'b_norm_3/AssignMovingAvg/ReadVariableOp�.b_norm_3/AssignMovingAvg_1/AssignSubVariableOp�)b_norm_3/AssignMovingAvg_1/ReadVariableOp�b_norm_3/ReadVariableOp�b_norm_3/ReadVariableOp_1�,b_norm_4/AssignMovingAvg/AssignSubVariableOp�'b_norm_4/AssignMovingAvg/ReadVariableOp�.b_norm_4/AssignMovingAvg_1/AssignSubVariableOp�)b_norm_4/AssignMovingAvg_1/ReadVariableOp�b_norm_4/ReadVariableOp�b_norm_4/ReadVariableOp_1�conv_1/BiasAdd/ReadVariableOp�conv_1/Conv2D/ReadVariableOp�conv_2/BiasAdd/ReadVariableOp�conv_2/Conv2D/ReadVariableOp�conv_3/BiasAdd/ReadVariableOp�conv_3/Conv2D/ReadVariableOp�conv_4/BiasAdd/ReadVariableOp�conv_4/Conv2D/ReadVariableOp�=input_batch_normalization/AssignMovingAvg/AssignSubVariableOp�8input_batch_normalization/AssignMovingAvg/ReadVariableOp�?input_batch_normalization/AssignMovingAvg_1/AssignSubVariableOp�:input_batch_normalization/AssignMovingAvg_1/ReadVariableOp�(input_batch_normalization/ReadVariableOp�*input_batch_normalization/ReadVariableOp_1�!input_conv/BiasAdd/ReadVariableOp� input_conv/Conv2D/ReadVariableOp�0output_class_distribution/BiasAdd/ReadVariableOp�/output_class_distribution/MatMul/ReadVariableOp�
 input_conv/Conv2D/ReadVariableOpReadVariableOp)input_conv_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02"
 input_conv/Conv2D/ReadVariableOp�
input_conv/Conv2DConv2Dinputs(input_conv/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:�����������*
paddingSAME*
strides
2
input_conv/Conv2D�
!input_conv/BiasAdd/ReadVariableOpReadVariableOp*input_conv_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02#
!input_conv/BiasAdd/ReadVariableOp�
input_conv/BiasAddBiasAddinput_conv/Conv2D:output:0)input_conv/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:�����������2
input_conv/BiasAdd�
&input_batch_normalization/LogicalAnd/xConst*
_output_shapes
: *
dtype0
*
value	B
 Z2(
&input_batch_normalization/LogicalAnd/x�
&input_batch_normalization/LogicalAnd/yConst*
_output_shapes
: *
dtype0
*
value	B
 Z2(
&input_batch_normalization/LogicalAnd/y�
$input_batch_normalization/LogicalAnd
LogicalAnd/input_batch_normalization/LogicalAnd/x:output:0/input_batch_normalization/LogicalAnd/y:output:0*
_output_shapes
: 2&
$input_batch_normalization/LogicalAnd�
(input_batch_normalization/ReadVariableOpReadVariableOp1input_batch_normalization_readvariableop_resource*
_output_shapes
:*
dtype02*
(input_batch_normalization/ReadVariableOp�
*input_batch_normalization/ReadVariableOp_1ReadVariableOp3input_batch_normalization_readvariableop_1_resource*
_output_shapes
:*
dtype02,
*input_batch_normalization/ReadVariableOp_1�
input_batch_normalization/ConstConst*
_output_shapes
: *
dtype0*
valueB 2!
input_batch_normalization/Const�
!input_batch_normalization/Const_1Const*
_output_shapes
: *
dtype0*
valueB 2#
!input_batch_normalization/Const_1�
*input_batch_normalization/FusedBatchNormV3FusedBatchNormV3input_conv/BiasAdd:output:00input_batch_normalization/ReadVariableOp:value:02input_batch_normalization/ReadVariableOp_1:value:0(input_batch_normalization/Const:output:0*input_batch_normalization/Const_1:output:0*
T0*
U0*M
_output_shapes;
9:�����������:::::*
epsilon%o�:2,
*input_batch_normalization/FusedBatchNormV3�
!input_batch_normalization/Const_2Const*
_output_shapes
: *
dtype0*
valueB
 *�p}?2#
!input_batch_normalization/Const_2�
/input_batch_normalization/AssignMovingAvg/sub/xConst*D
_class:
86loc:@input_batch_normalization/AssignMovingAvg/4353835*
_output_shapes
: *
dtype0*
valueB
 *  �?21
/input_batch_normalization/AssignMovingAvg/sub/x�
-input_batch_normalization/AssignMovingAvg/subSub8input_batch_normalization/AssignMovingAvg/sub/x:output:0*input_batch_normalization/Const_2:output:0*
T0*D
_class:
86loc:@input_batch_normalization/AssignMovingAvg/4353835*
_output_shapes
: 2/
-input_batch_normalization/AssignMovingAvg/sub�
8input_batch_normalization/AssignMovingAvg/ReadVariableOpReadVariableOp1input_batch_normalization_assignmovingavg_4353835*
_output_shapes
:*
dtype02:
8input_batch_normalization/AssignMovingAvg/ReadVariableOp�
/input_batch_normalization/AssignMovingAvg/sub_1Sub@input_batch_normalization/AssignMovingAvg/ReadVariableOp:value:07input_batch_normalization/FusedBatchNormV3:batch_mean:0*
T0*D
_class:
86loc:@input_batch_normalization/AssignMovingAvg/4353835*
_output_shapes
:21
/input_batch_normalization/AssignMovingAvg/sub_1�
-input_batch_normalization/AssignMovingAvg/mulMul3input_batch_normalization/AssignMovingAvg/sub_1:z:01input_batch_normalization/AssignMovingAvg/sub:z:0*
T0*D
_class:
86loc:@input_batch_normalization/AssignMovingAvg/4353835*
_output_shapes
:2/
-input_batch_normalization/AssignMovingAvg/mul�
=input_batch_normalization/AssignMovingAvg/AssignSubVariableOpAssignSubVariableOp1input_batch_normalization_assignmovingavg_43538351input_batch_normalization/AssignMovingAvg/mul:z:09^input_batch_normalization/AssignMovingAvg/ReadVariableOp*D
_class:
86loc:@input_batch_normalization/AssignMovingAvg/4353835*
_output_shapes
 *
dtype02?
=input_batch_normalization/AssignMovingAvg/AssignSubVariableOp�
1input_batch_normalization/AssignMovingAvg_1/sub/xConst*F
_class<
:8loc:@input_batch_normalization/AssignMovingAvg_1/4353842*
_output_shapes
: *
dtype0*
valueB
 *  �?23
1input_batch_normalization/AssignMovingAvg_1/sub/x�
/input_batch_normalization/AssignMovingAvg_1/subSub:input_batch_normalization/AssignMovingAvg_1/sub/x:output:0*input_batch_normalization/Const_2:output:0*
T0*F
_class<
:8loc:@input_batch_normalization/AssignMovingAvg_1/4353842*
_output_shapes
: 21
/input_batch_normalization/AssignMovingAvg_1/sub�
:input_batch_normalization/AssignMovingAvg_1/ReadVariableOpReadVariableOp3input_batch_normalization_assignmovingavg_1_4353842*
_output_shapes
:*
dtype02<
:input_batch_normalization/AssignMovingAvg_1/ReadVariableOp�
1input_batch_normalization/AssignMovingAvg_1/sub_1SubBinput_batch_normalization/AssignMovingAvg_1/ReadVariableOp:value:0;input_batch_normalization/FusedBatchNormV3:batch_variance:0*
T0*F
_class<
:8loc:@input_batch_normalization/AssignMovingAvg_1/4353842*
_output_shapes
:23
1input_batch_normalization/AssignMovingAvg_1/sub_1�
/input_batch_normalization/AssignMovingAvg_1/mulMul5input_batch_normalization/AssignMovingAvg_1/sub_1:z:03input_batch_normalization/AssignMovingAvg_1/sub:z:0*
T0*F
_class<
:8loc:@input_batch_normalization/AssignMovingAvg_1/4353842*
_output_shapes
:21
/input_batch_normalization/AssignMovingAvg_1/mul�
?input_batch_normalization/AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOp3input_batch_normalization_assignmovingavg_1_43538423input_batch_normalization/AssignMovingAvg_1/mul:z:0;^input_batch_normalization/AssignMovingAvg_1/ReadVariableOp*F
_class<
:8loc:@input_batch_normalization/AssignMovingAvg_1/4353842*
_output_shapes
 *
dtype02A
?input_batch_normalization/AssignMovingAvg_1/AssignSubVariableOp�
input_RELU/ReluRelu.input_batch_normalization/FusedBatchNormV3:y:0*
T0*1
_output_shapes
:�����������2
input_RELU/Relu�
max_pooling2d_85/MaxPoolMaxPoolinput_RELU/Relu:activations:0*/
_output_shapes
:���������@@*
ksize
*
paddingVALID*
strides
2
max_pooling2d_85/MaxPoolw
dropout_42/dropout/rateConst*
_output_shapes
: *
dtype0*
valueB
 *33�>2
dropout_42/dropout/rate�
dropout_42/dropout/ShapeShape!max_pooling2d_85/MaxPool:output:0*
T0*
_output_shapes
:2
dropout_42/dropout/Shape�
%dropout_42/dropout/random_uniform/minConst*
_output_shapes
: *
dtype0*
valueB
 *    2'
%dropout_42/dropout/random_uniform/min�
%dropout_42/dropout/random_uniform/maxConst*
_output_shapes
: *
dtype0*
valueB
 *  �?2'
%dropout_42/dropout/random_uniform/max�
/dropout_42/dropout/random_uniform/RandomUniformRandomUniform!dropout_42/dropout/Shape:output:0*
T0*/
_output_shapes
:���������@@*
dtype021
/dropout_42/dropout/random_uniform/RandomUniform�
%dropout_42/dropout/random_uniform/subSub.dropout_42/dropout/random_uniform/max:output:0.dropout_42/dropout/random_uniform/min:output:0*
T0*
_output_shapes
: 2'
%dropout_42/dropout/random_uniform/sub�
%dropout_42/dropout/random_uniform/mulMul8dropout_42/dropout/random_uniform/RandomUniform:output:0)dropout_42/dropout/random_uniform/sub:z:0*
T0*/
_output_shapes
:���������@@2'
%dropout_42/dropout/random_uniform/mul�
!dropout_42/dropout/random_uniformAdd)dropout_42/dropout/random_uniform/mul:z:0.dropout_42/dropout/random_uniform/min:output:0*
T0*/
_output_shapes
:���������@@2#
!dropout_42/dropout/random_uniformy
dropout_42/dropout/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?2
dropout_42/dropout/sub/x�
dropout_42/dropout/subSub!dropout_42/dropout/sub/x:output:0 dropout_42/dropout/rate:output:0*
T0*
_output_shapes
: 2
dropout_42/dropout/sub�
dropout_42/dropout/truediv/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?2
dropout_42/dropout/truediv/x�
dropout_42/dropout/truedivRealDiv%dropout_42/dropout/truediv/x:output:0dropout_42/dropout/sub:z:0*
T0*
_output_shapes
: 2
dropout_42/dropout/truediv�
dropout_42/dropout/GreaterEqualGreaterEqual%dropout_42/dropout/random_uniform:z:0 dropout_42/dropout/rate:output:0*
T0*/
_output_shapes
:���������@@2!
dropout_42/dropout/GreaterEqual�
dropout_42/dropout/mulMul!max_pooling2d_85/MaxPool:output:0dropout_42/dropout/truediv:z:0*
T0*/
_output_shapes
:���������@@2
dropout_42/dropout/mul�
dropout_42/dropout/CastCast#dropout_42/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:���������@@2
dropout_42/dropout/Cast�
dropout_42/dropout/mul_1Muldropout_42/dropout/mul:z:0dropout_42/dropout/Cast:y:0*
T0*/
_output_shapes
:���������@@2
dropout_42/dropout/mul_1�
conv_1/Conv2D/ReadVariableOpReadVariableOp%conv_1_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02
conv_1/Conv2D/ReadVariableOp�
conv_1/Conv2DConv2Ddropout_42/dropout/mul_1:z:0$conv_1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@@*
paddingSAME*
strides
2
conv_1/Conv2D�
conv_1/BiasAdd/ReadVariableOpReadVariableOp&conv_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02
conv_1/BiasAdd/ReadVariableOp�
conv_1/BiasAddBiasAddconv_1/Conv2D:output:0%conv_1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@@2
conv_1/BiasAddp
b_norm_1/LogicalAnd/xConst*
_output_shapes
: *
dtype0
*
value	B
 Z2
b_norm_1/LogicalAnd/xp
b_norm_1/LogicalAnd/yConst*
_output_shapes
: *
dtype0
*
value	B
 Z2
b_norm_1/LogicalAnd/y�
b_norm_1/LogicalAnd
LogicalAndb_norm_1/LogicalAnd/x:output:0b_norm_1/LogicalAnd/y:output:0*
_output_shapes
: 2
b_norm_1/LogicalAnd�
b_norm_1/ReadVariableOpReadVariableOp b_norm_1_readvariableop_resource*
_output_shapes
:*
dtype02
b_norm_1/ReadVariableOp�
b_norm_1/ReadVariableOp_1ReadVariableOp"b_norm_1_readvariableop_1_resource*
_output_shapes
:*
dtype02
b_norm_1/ReadVariableOp_1c
b_norm_1/ConstConst*
_output_shapes
: *
dtype0*
valueB 2
b_norm_1/Constg
b_norm_1/Const_1Const*
_output_shapes
: *
dtype0*
valueB 2
b_norm_1/Const_1�
b_norm_1/FusedBatchNormV3FusedBatchNormV3conv_1/BiasAdd:output:0b_norm_1/ReadVariableOp:value:0!b_norm_1/ReadVariableOp_1:value:0b_norm_1/Const:output:0b_norm_1/Const_1:output:0*
T0*
U0*K
_output_shapes9
7:���������@@:::::*
epsilon%o�:2
b_norm_1/FusedBatchNormV3i
b_norm_1/Const_2Const*
_output_shapes
: *
dtype0*
valueB
 *�p}?2
b_norm_1/Const_2�
b_norm_1/AssignMovingAvg/sub/xConst*3
_class)
'%loc:@b_norm_1/AssignMovingAvg/4353889*
_output_shapes
: *
dtype0*
valueB
 *  �?2 
b_norm_1/AssignMovingAvg/sub/x�
b_norm_1/AssignMovingAvg/subSub'b_norm_1/AssignMovingAvg/sub/x:output:0b_norm_1/Const_2:output:0*
T0*3
_class)
'%loc:@b_norm_1/AssignMovingAvg/4353889*
_output_shapes
: 2
b_norm_1/AssignMovingAvg/sub�
'b_norm_1/AssignMovingAvg/ReadVariableOpReadVariableOp b_norm_1_assignmovingavg_4353889*
_output_shapes
:*
dtype02)
'b_norm_1/AssignMovingAvg/ReadVariableOp�
b_norm_1/AssignMovingAvg/sub_1Sub/b_norm_1/AssignMovingAvg/ReadVariableOp:value:0&b_norm_1/FusedBatchNormV3:batch_mean:0*
T0*3
_class)
'%loc:@b_norm_1/AssignMovingAvg/4353889*
_output_shapes
:2 
b_norm_1/AssignMovingAvg/sub_1�
b_norm_1/AssignMovingAvg/mulMul"b_norm_1/AssignMovingAvg/sub_1:z:0 b_norm_1/AssignMovingAvg/sub:z:0*
T0*3
_class)
'%loc:@b_norm_1/AssignMovingAvg/4353889*
_output_shapes
:2
b_norm_1/AssignMovingAvg/mul�
,b_norm_1/AssignMovingAvg/AssignSubVariableOpAssignSubVariableOp b_norm_1_assignmovingavg_4353889 b_norm_1/AssignMovingAvg/mul:z:0(^b_norm_1/AssignMovingAvg/ReadVariableOp*3
_class)
'%loc:@b_norm_1/AssignMovingAvg/4353889*
_output_shapes
 *
dtype02.
,b_norm_1/AssignMovingAvg/AssignSubVariableOp�
 b_norm_1/AssignMovingAvg_1/sub/xConst*5
_class+
)'loc:@b_norm_1/AssignMovingAvg_1/4353896*
_output_shapes
: *
dtype0*
valueB
 *  �?2"
 b_norm_1/AssignMovingAvg_1/sub/x�
b_norm_1/AssignMovingAvg_1/subSub)b_norm_1/AssignMovingAvg_1/sub/x:output:0b_norm_1/Const_2:output:0*
T0*5
_class+
)'loc:@b_norm_1/AssignMovingAvg_1/4353896*
_output_shapes
: 2 
b_norm_1/AssignMovingAvg_1/sub�
)b_norm_1/AssignMovingAvg_1/ReadVariableOpReadVariableOp"b_norm_1_assignmovingavg_1_4353896*
_output_shapes
:*
dtype02+
)b_norm_1/AssignMovingAvg_1/ReadVariableOp�
 b_norm_1/AssignMovingAvg_1/sub_1Sub1b_norm_1/AssignMovingAvg_1/ReadVariableOp:value:0*b_norm_1/FusedBatchNormV3:batch_variance:0*
T0*5
_class+
)'loc:@b_norm_1/AssignMovingAvg_1/4353896*
_output_shapes
:2"
 b_norm_1/AssignMovingAvg_1/sub_1�
b_norm_1/AssignMovingAvg_1/mulMul$b_norm_1/AssignMovingAvg_1/sub_1:z:0"b_norm_1/AssignMovingAvg_1/sub:z:0*
T0*5
_class+
)'loc:@b_norm_1/AssignMovingAvg_1/4353896*
_output_shapes
:2 
b_norm_1/AssignMovingAvg_1/mul�
.b_norm_1/AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOp"b_norm_1_assignmovingavg_1_4353896"b_norm_1/AssignMovingAvg_1/mul:z:0*^b_norm_1/AssignMovingAvg_1/ReadVariableOp*5
_class+
)'loc:@b_norm_1/AssignMovingAvg_1/4353896*
_output_shapes
 *
dtype020
.b_norm_1/AssignMovingAvg_1/AssignSubVariableOp{
RELU_1/ReluRelub_norm_1/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:���������@@2
RELU_1/Relu�
max_pooling2d_81/MaxPoolMaxPoolRELU_1/Relu:activations:0*/
_output_shapes
:���������  *
ksize
*
paddingVALID*
strides
2
max_pooling2d_81/MaxPool�
conv_2/Conv2D/ReadVariableOpReadVariableOp%conv_2_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02
conv_2/Conv2D/ReadVariableOp�
conv_2/Conv2DConv2D!max_pooling2d_81/MaxPool:output:0$conv_2/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������   *
paddingSAME*
strides
2
conv_2/Conv2D�
conv_2/BiasAdd/ReadVariableOpReadVariableOp&conv_2_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02
conv_2/BiasAdd/ReadVariableOp�
conv_2/BiasAddBiasAddconv_2/Conv2D:output:0%conv_2/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������   2
conv_2/BiasAddp
b_norm_2/LogicalAnd/xConst*
_output_shapes
: *
dtype0
*
value	B
 Z2
b_norm_2/LogicalAnd/xp
b_norm_2/LogicalAnd/yConst*
_output_shapes
: *
dtype0
*
value	B
 Z2
b_norm_2/LogicalAnd/y�
b_norm_2/LogicalAnd
LogicalAndb_norm_2/LogicalAnd/x:output:0b_norm_2/LogicalAnd/y:output:0*
_output_shapes
: 2
b_norm_2/LogicalAnd�
b_norm_2/ReadVariableOpReadVariableOp b_norm_2_readvariableop_resource*
_output_shapes
: *
dtype02
b_norm_2/ReadVariableOp�
b_norm_2/ReadVariableOp_1ReadVariableOp"b_norm_2_readvariableop_1_resource*
_output_shapes
: *
dtype02
b_norm_2/ReadVariableOp_1c
b_norm_2/ConstConst*
_output_shapes
: *
dtype0*
valueB 2
b_norm_2/Constg
b_norm_2/Const_1Const*
_output_shapes
: *
dtype0*
valueB 2
b_norm_2/Const_1�
b_norm_2/FusedBatchNormV3FusedBatchNormV3conv_2/BiasAdd:output:0b_norm_2/ReadVariableOp:value:0!b_norm_2/ReadVariableOp_1:value:0b_norm_2/Const:output:0b_norm_2/Const_1:output:0*
T0*
U0*K
_output_shapes9
7:���������   : : : : :*
epsilon%o�:2
b_norm_2/FusedBatchNormV3i
b_norm_2/Const_2Const*
_output_shapes
: *
dtype0*
valueB
 *�p}?2
b_norm_2/Const_2�
b_norm_2/AssignMovingAvg/sub/xConst*3
_class)
'%loc:@b_norm_2/AssignMovingAvg/4353927*
_output_shapes
: *
dtype0*
valueB
 *  �?2 
b_norm_2/AssignMovingAvg/sub/x�
b_norm_2/AssignMovingAvg/subSub'b_norm_2/AssignMovingAvg/sub/x:output:0b_norm_2/Const_2:output:0*
T0*3
_class)
'%loc:@b_norm_2/AssignMovingAvg/4353927*
_output_shapes
: 2
b_norm_2/AssignMovingAvg/sub�
'b_norm_2/AssignMovingAvg/ReadVariableOpReadVariableOp b_norm_2_assignmovingavg_4353927*
_output_shapes
: *
dtype02)
'b_norm_2/AssignMovingAvg/ReadVariableOp�
b_norm_2/AssignMovingAvg/sub_1Sub/b_norm_2/AssignMovingAvg/ReadVariableOp:value:0&b_norm_2/FusedBatchNormV3:batch_mean:0*
T0*3
_class)
'%loc:@b_norm_2/AssignMovingAvg/4353927*
_output_shapes
: 2 
b_norm_2/AssignMovingAvg/sub_1�
b_norm_2/AssignMovingAvg/mulMul"b_norm_2/AssignMovingAvg/sub_1:z:0 b_norm_2/AssignMovingAvg/sub:z:0*
T0*3
_class)
'%loc:@b_norm_2/AssignMovingAvg/4353927*
_output_shapes
: 2
b_norm_2/AssignMovingAvg/mul�
,b_norm_2/AssignMovingAvg/AssignSubVariableOpAssignSubVariableOp b_norm_2_assignmovingavg_4353927 b_norm_2/AssignMovingAvg/mul:z:0(^b_norm_2/AssignMovingAvg/ReadVariableOp*3
_class)
'%loc:@b_norm_2/AssignMovingAvg/4353927*
_output_shapes
 *
dtype02.
,b_norm_2/AssignMovingAvg/AssignSubVariableOp�
 b_norm_2/AssignMovingAvg_1/sub/xConst*5
_class+
)'loc:@b_norm_2/AssignMovingAvg_1/4353934*
_output_shapes
: *
dtype0*
valueB
 *  �?2"
 b_norm_2/AssignMovingAvg_1/sub/x�
b_norm_2/AssignMovingAvg_1/subSub)b_norm_2/AssignMovingAvg_1/sub/x:output:0b_norm_2/Const_2:output:0*
T0*5
_class+
)'loc:@b_norm_2/AssignMovingAvg_1/4353934*
_output_shapes
: 2 
b_norm_2/AssignMovingAvg_1/sub�
)b_norm_2/AssignMovingAvg_1/ReadVariableOpReadVariableOp"b_norm_2_assignmovingavg_1_4353934*
_output_shapes
: *
dtype02+
)b_norm_2/AssignMovingAvg_1/ReadVariableOp�
 b_norm_2/AssignMovingAvg_1/sub_1Sub1b_norm_2/AssignMovingAvg_1/ReadVariableOp:value:0*b_norm_2/FusedBatchNormV3:batch_variance:0*
T0*5
_class+
)'loc:@b_norm_2/AssignMovingAvg_1/4353934*
_output_shapes
: 2"
 b_norm_2/AssignMovingAvg_1/sub_1�
b_norm_2/AssignMovingAvg_1/mulMul$b_norm_2/AssignMovingAvg_1/sub_1:z:0"b_norm_2/AssignMovingAvg_1/sub:z:0*
T0*5
_class+
)'loc:@b_norm_2/AssignMovingAvg_1/4353934*
_output_shapes
: 2 
b_norm_2/AssignMovingAvg_1/mul�
.b_norm_2/AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOp"b_norm_2_assignmovingavg_1_4353934"b_norm_2/AssignMovingAvg_1/mul:z:0*^b_norm_2/AssignMovingAvg_1/ReadVariableOp*5
_class+
)'loc:@b_norm_2/AssignMovingAvg_1/4353934*
_output_shapes
 *
dtype020
.b_norm_2/AssignMovingAvg_1/AssignSubVariableOp{
RELU_2/ReluRelub_norm_2/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:���������   2
RELU_2/Relu�
max_pooling2d_82/MaxPoolMaxPoolRELU_2/Relu:activations:0*/
_output_shapes
:��������� *
ksize
*
paddingVALID*
strides
2
max_pooling2d_82/MaxPool�
conv_3/Conv2D/ReadVariableOpReadVariableOp%conv_3_conv2d_readvariableop_resource*&
_output_shapes
: @*
dtype02
conv_3/Conv2D/ReadVariableOp�
conv_3/Conv2DConv2D!max_pooling2d_82/MaxPool:output:0$conv_3/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@*
paddingSAME*
strides
2
conv_3/Conv2D�
conv_3/BiasAdd/ReadVariableOpReadVariableOp&conv_3_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
conv_3/BiasAdd/ReadVariableOp�
conv_3/BiasAddBiasAddconv_3/Conv2D:output:0%conv_3/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@2
conv_3/BiasAddp
b_norm_3/LogicalAnd/xConst*
_output_shapes
: *
dtype0
*
value	B
 Z2
b_norm_3/LogicalAnd/xp
b_norm_3/LogicalAnd/yConst*
_output_shapes
: *
dtype0
*
value	B
 Z2
b_norm_3/LogicalAnd/y�
b_norm_3/LogicalAnd
LogicalAndb_norm_3/LogicalAnd/x:output:0b_norm_3/LogicalAnd/y:output:0*
_output_shapes
: 2
b_norm_3/LogicalAnd�
b_norm_3/ReadVariableOpReadVariableOp b_norm_3_readvariableop_resource*
_output_shapes
:@*
dtype02
b_norm_3/ReadVariableOp�
b_norm_3/ReadVariableOp_1ReadVariableOp"b_norm_3_readvariableop_1_resource*
_output_shapes
:@*
dtype02
b_norm_3/ReadVariableOp_1c
b_norm_3/ConstConst*
_output_shapes
: *
dtype0*
valueB 2
b_norm_3/Constg
b_norm_3/Const_1Const*
_output_shapes
: *
dtype0*
valueB 2
b_norm_3/Const_1�
b_norm_3/FusedBatchNormV3FusedBatchNormV3conv_3/BiasAdd:output:0b_norm_3/ReadVariableOp:value:0!b_norm_3/ReadVariableOp_1:value:0b_norm_3/Const:output:0b_norm_3/Const_1:output:0*
T0*
U0*K
_output_shapes9
7:���������@:@:@:@:@:*
epsilon%o�:2
b_norm_3/FusedBatchNormV3i
b_norm_3/Const_2Const*
_output_shapes
: *
dtype0*
valueB
 *�p}?2
b_norm_3/Const_2�
b_norm_3/AssignMovingAvg/sub/xConst*3
_class)
'%loc:@b_norm_3/AssignMovingAvg/4353965*
_output_shapes
: *
dtype0*
valueB
 *  �?2 
b_norm_3/AssignMovingAvg/sub/x�
b_norm_3/AssignMovingAvg/subSub'b_norm_3/AssignMovingAvg/sub/x:output:0b_norm_3/Const_2:output:0*
T0*3
_class)
'%loc:@b_norm_3/AssignMovingAvg/4353965*
_output_shapes
: 2
b_norm_3/AssignMovingAvg/sub�
'b_norm_3/AssignMovingAvg/ReadVariableOpReadVariableOp b_norm_3_assignmovingavg_4353965*
_output_shapes
:@*
dtype02)
'b_norm_3/AssignMovingAvg/ReadVariableOp�
b_norm_3/AssignMovingAvg/sub_1Sub/b_norm_3/AssignMovingAvg/ReadVariableOp:value:0&b_norm_3/FusedBatchNormV3:batch_mean:0*
T0*3
_class)
'%loc:@b_norm_3/AssignMovingAvg/4353965*
_output_shapes
:@2 
b_norm_3/AssignMovingAvg/sub_1�
b_norm_3/AssignMovingAvg/mulMul"b_norm_3/AssignMovingAvg/sub_1:z:0 b_norm_3/AssignMovingAvg/sub:z:0*
T0*3
_class)
'%loc:@b_norm_3/AssignMovingAvg/4353965*
_output_shapes
:@2
b_norm_3/AssignMovingAvg/mul�
,b_norm_3/AssignMovingAvg/AssignSubVariableOpAssignSubVariableOp b_norm_3_assignmovingavg_4353965 b_norm_3/AssignMovingAvg/mul:z:0(^b_norm_3/AssignMovingAvg/ReadVariableOp*3
_class)
'%loc:@b_norm_3/AssignMovingAvg/4353965*
_output_shapes
 *
dtype02.
,b_norm_3/AssignMovingAvg/AssignSubVariableOp�
 b_norm_3/AssignMovingAvg_1/sub/xConst*5
_class+
)'loc:@b_norm_3/AssignMovingAvg_1/4353972*
_output_shapes
: *
dtype0*
valueB
 *  �?2"
 b_norm_3/AssignMovingAvg_1/sub/x�
b_norm_3/AssignMovingAvg_1/subSub)b_norm_3/AssignMovingAvg_1/sub/x:output:0b_norm_3/Const_2:output:0*
T0*5
_class+
)'loc:@b_norm_3/AssignMovingAvg_1/4353972*
_output_shapes
: 2 
b_norm_3/AssignMovingAvg_1/sub�
)b_norm_3/AssignMovingAvg_1/ReadVariableOpReadVariableOp"b_norm_3_assignmovingavg_1_4353972*
_output_shapes
:@*
dtype02+
)b_norm_3/AssignMovingAvg_1/ReadVariableOp�
 b_norm_3/AssignMovingAvg_1/sub_1Sub1b_norm_3/AssignMovingAvg_1/ReadVariableOp:value:0*b_norm_3/FusedBatchNormV3:batch_variance:0*
T0*5
_class+
)'loc:@b_norm_3/AssignMovingAvg_1/4353972*
_output_shapes
:@2"
 b_norm_3/AssignMovingAvg_1/sub_1�
b_norm_3/AssignMovingAvg_1/mulMul$b_norm_3/AssignMovingAvg_1/sub_1:z:0"b_norm_3/AssignMovingAvg_1/sub:z:0*
T0*5
_class+
)'loc:@b_norm_3/AssignMovingAvg_1/4353972*
_output_shapes
:@2 
b_norm_3/AssignMovingAvg_1/mul�
.b_norm_3/AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOp"b_norm_3_assignmovingavg_1_4353972"b_norm_3/AssignMovingAvg_1/mul:z:0*^b_norm_3/AssignMovingAvg_1/ReadVariableOp*5
_class+
)'loc:@b_norm_3/AssignMovingAvg_1/4353972*
_output_shapes
 *
dtype020
.b_norm_3/AssignMovingAvg_1/AssignSubVariableOp{
RELU_3/ReluRelub_norm_3/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:���������@2
RELU_3/Relu�
max_pooling2d_83/MaxPoolMaxPoolRELU_3/Relu:activations:0*/
_output_shapes
:���������@*
ksize
*
paddingVALID*
strides
2
max_pooling2d_83/MaxPool�
conv_4/Conv2D/ReadVariableOpReadVariableOp%conv_4_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02
conv_4/Conv2D/ReadVariableOp�
conv_4/Conv2DConv2D!max_pooling2d_83/MaxPool:output:0$conv_4/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@*
paddingSAME*
strides
2
conv_4/Conv2D�
conv_4/BiasAdd/ReadVariableOpReadVariableOp&conv_4_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
conv_4/BiasAdd/ReadVariableOp�
conv_4/BiasAddBiasAddconv_4/Conv2D:output:0%conv_4/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@2
conv_4/BiasAddp
b_norm_4/LogicalAnd/xConst*
_output_shapes
: *
dtype0
*
value	B
 Z2
b_norm_4/LogicalAnd/xp
b_norm_4/LogicalAnd/yConst*
_output_shapes
: *
dtype0
*
value	B
 Z2
b_norm_4/LogicalAnd/y�
b_norm_4/LogicalAnd
LogicalAndb_norm_4/LogicalAnd/x:output:0b_norm_4/LogicalAnd/y:output:0*
_output_shapes
: 2
b_norm_4/LogicalAnd�
b_norm_4/ReadVariableOpReadVariableOp b_norm_4_readvariableop_resource*
_output_shapes
:@*
dtype02
b_norm_4/ReadVariableOp�
b_norm_4/ReadVariableOp_1ReadVariableOp"b_norm_4_readvariableop_1_resource*
_output_shapes
:@*
dtype02
b_norm_4/ReadVariableOp_1c
b_norm_4/ConstConst*
_output_shapes
: *
dtype0*
valueB 2
b_norm_4/Constg
b_norm_4/Const_1Const*
_output_shapes
: *
dtype0*
valueB 2
b_norm_4/Const_1�
b_norm_4/FusedBatchNormV3FusedBatchNormV3conv_4/BiasAdd:output:0b_norm_4/ReadVariableOp:value:0!b_norm_4/ReadVariableOp_1:value:0b_norm_4/Const:output:0b_norm_4/Const_1:output:0*
T0*
U0*K
_output_shapes9
7:���������@:@:@:@:@:*
epsilon%o�:2
b_norm_4/FusedBatchNormV3i
b_norm_4/Const_2Const*
_output_shapes
: *
dtype0*
valueB
 *�p}?2
b_norm_4/Const_2�
b_norm_4/AssignMovingAvg/sub/xConst*3
_class)
'%loc:@b_norm_4/AssignMovingAvg/4354003*
_output_shapes
: *
dtype0*
valueB
 *  �?2 
b_norm_4/AssignMovingAvg/sub/x�
b_norm_4/AssignMovingAvg/subSub'b_norm_4/AssignMovingAvg/sub/x:output:0b_norm_4/Const_2:output:0*
T0*3
_class)
'%loc:@b_norm_4/AssignMovingAvg/4354003*
_output_shapes
: 2
b_norm_4/AssignMovingAvg/sub�
'b_norm_4/AssignMovingAvg/ReadVariableOpReadVariableOp b_norm_4_assignmovingavg_4354003*
_output_shapes
:@*
dtype02)
'b_norm_4/AssignMovingAvg/ReadVariableOp�
b_norm_4/AssignMovingAvg/sub_1Sub/b_norm_4/AssignMovingAvg/ReadVariableOp:value:0&b_norm_4/FusedBatchNormV3:batch_mean:0*
T0*3
_class)
'%loc:@b_norm_4/AssignMovingAvg/4354003*
_output_shapes
:@2 
b_norm_4/AssignMovingAvg/sub_1�
b_norm_4/AssignMovingAvg/mulMul"b_norm_4/AssignMovingAvg/sub_1:z:0 b_norm_4/AssignMovingAvg/sub:z:0*
T0*3
_class)
'%loc:@b_norm_4/AssignMovingAvg/4354003*
_output_shapes
:@2
b_norm_4/AssignMovingAvg/mul�
,b_norm_4/AssignMovingAvg/AssignSubVariableOpAssignSubVariableOp b_norm_4_assignmovingavg_4354003 b_norm_4/AssignMovingAvg/mul:z:0(^b_norm_4/AssignMovingAvg/ReadVariableOp*3
_class)
'%loc:@b_norm_4/AssignMovingAvg/4354003*
_output_shapes
 *
dtype02.
,b_norm_4/AssignMovingAvg/AssignSubVariableOp�
 b_norm_4/AssignMovingAvg_1/sub/xConst*5
_class+
)'loc:@b_norm_4/AssignMovingAvg_1/4354010*
_output_shapes
: *
dtype0*
valueB
 *  �?2"
 b_norm_4/AssignMovingAvg_1/sub/x�
b_norm_4/AssignMovingAvg_1/subSub)b_norm_4/AssignMovingAvg_1/sub/x:output:0b_norm_4/Const_2:output:0*
T0*5
_class+
)'loc:@b_norm_4/AssignMovingAvg_1/4354010*
_output_shapes
: 2 
b_norm_4/AssignMovingAvg_1/sub�
)b_norm_4/AssignMovingAvg_1/ReadVariableOpReadVariableOp"b_norm_4_assignmovingavg_1_4354010*
_output_shapes
:@*
dtype02+
)b_norm_4/AssignMovingAvg_1/ReadVariableOp�
 b_norm_4/AssignMovingAvg_1/sub_1Sub1b_norm_4/AssignMovingAvg_1/ReadVariableOp:value:0*b_norm_4/FusedBatchNormV3:batch_variance:0*
T0*5
_class+
)'loc:@b_norm_4/AssignMovingAvg_1/4354010*
_output_shapes
:@2"
 b_norm_4/AssignMovingAvg_1/sub_1�
b_norm_4/AssignMovingAvg_1/mulMul$b_norm_4/AssignMovingAvg_1/sub_1:z:0"b_norm_4/AssignMovingAvg_1/sub:z:0*
T0*5
_class+
)'loc:@b_norm_4/AssignMovingAvg_1/4354010*
_output_shapes
:@2 
b_norm_4/AssignMovingAvg_1/mul�
.b_norm_4/AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOp"b_norm_4_assignmovingavg_1_4354010"b_norm_4/AssignMovingAvg_1/mul:z:0*^b_norm_4/AssignMovingAvg_1/ReadVariableOp*5
_class+
)'loc:@b_norm_4/AssignMovingAvg_1/4354010*
_output_shapes
 *
dtype020
.b_norm_4/AssignMovingAvg_1/AssignSubVariableOp{
RELU_4/ReluRelub_norm_4/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:���������@2
RELU_4/Relu�
max_pooling2d_84/MaxPoolMaxPoolRELU_4/Relu:activations:0*/
_output_shapes
:���������@*
ksize
*
paddingVALID*
strides
2
max_pooling2d_84/MaxPoolu
flatten_21/ConstConst*
_output_shapes
:*
dtype0*
valueB"����   2
flatten_21/Const�
flatten_21/ReshapeReshape!max_pooling2d_84/MaxPool:output:0flatten_21/Const:output:0*
T0*(
_output_shapes
:����������2
flatten_21/Reshapew
dropout_43/dropout/rateConst*
_output_shapes
: *
dtype0*
valueB
 *33�>2
dropout_43/dropout/rate
dropout_43/dropout/ShapeShapeflatten_21/Reshape:output:0*
T0*
_output_shapes
:2
dropout_43/dropout/Shape�
%dropout_43/dropout/random_uniform/minConst*
_output_shapes
: *
dtype0*
valueB
 *    2'
%dropout_43/dropout/random_uniform/min�
%dropout_43/dropout/random_uniform/maxConst*
_output_shapes
: *
dtype0*
valueB
 *  �?2'
%dropout_43/dropout/random_uniform/max�
/dropout_43/dropout/random_uniform/RandomUniformRandomUniform!dropout_43/dropout/Shape:output:0*
T0*(
_output_shapes
:����������*
dtype021
/dropout_43/dropout/random_uniform/RandomUniform�
%dropout_43/dropout/random_uniform/subSub.dropout_43/dropout/random_uniform/max:output:0.dropout_43/dropout/random_uniform/min:output:0*
T0*
_output_shapes
: 2'
%dropout_43/dropout/random_uniform/sub�
%dropout_43/dropout/random_uniform/mulMul8dropout_43/dropout/random_uniform/RandomUniform:output:0)dropout_43/dropout/random_uniform/sub:z:0*
T0*(
_output_shapes
:����������2'
%dropout_43/dropout/random_uniform/mul�
!dropout_43/dropout/random_uniformAdd)dropout_43/dropout/random_uniform/mul:z:0.dropout_43/dropout/random_uniform/min:output:0*
T0*(
_output_shapes
:����������2#
!dropout_43/dropout/random_uniformy
dropout_43/dropout/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?2
dropout_43/dropout/sub/x�
dropout_43/dropout/subSub!dropout_43/dropout/sub/x:output:0 dropout_43/dropout/rate:output:0*
T0*
_output_shapes
: 2
dropout_43/dropout/sub�
dropout_43/dropout/truediv/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?2
dropout_43/dropout/truediv/x�
dropout_43/dropout/truedivRealDiv%dropout_43/dropout/truediv/x:output:0dropout_43/dropout/sub:z:0*
T0*
_output_shapes
: 2
dropout_43/dropout/truediv�
dropout_43/dropout/GreaterEqualGreaterEqual%dropout_43/dropout/random_uniform:z:0 dropout_43/dropout/rate:output:0*
T0*(
_output_shapes
:����������2!
dropout_43/dropout/GreaterEqual�
dropout_43/dropout/mulMulflatten_21/Reshape:output:0dropout_43/dropout/truediv:z:0*
T0*(
_output_shapes
:����������2
dropout_43/dropout/mul�
dropout_43/dropout/CastCast#dropout_43/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:����������2
dropout_43/dropout/Cast�
dropout_43/dropout/mul_1Muldropout_43/dropout/mul:z:0dropout_43/dropout/Cast:y:0*
T0*(
_output_shapes
:����������2
dropout_43/dropout/mul_1�
/output_class_distribution/MatMul/ReadVariableOpReadVariableOp8output_class_distribution_matmul_readvariableop_resource* 
_output_shapes
:
��*
dtype021
/output_class_distribution/MatMul/ReadVariableOp�
 output_class_distribution/MatMulMatMuldropout_43/dropout/mul_1:z:07output_class_distribution/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2"
 output_class_distribution/MatMul�
0output_class_distribution/BiasAdd/ReadVariableOpReadVariableOp9output_class_distribution_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype022
0output_class_distribution/BiasAdd/ReadVariableOp�
!output_class_distribution/BiasAddBiasAdd*output_class_distribution/MatMul:product:08output_class_distribution/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2#
!output_class_distribution/BiasAdd�
IdentityIdentity*output_class_distribution/BiasAdd:output:0-^b_norm_1/AssignMovingAvg/AssignSubVariableOp(^b_norm_1/AssignMovingAvg/ReadVariableOp/^b_norm_1/AssignMovingAvg_1/AssignSubVariableOp*^b_norm_1/AssignMovingAvg_1/ReadVariableOp^b_norm_1/ReadVariableOp^b_norm_1/ReadVariableOp_1-^b_norm_2/AssignMovingAvg/AssignSubVariableOp(^b_norm_2/AssignMovingAvg/ReadVariableOp/^b_norm_2/AssignMovingAvg_1/AssignSubVariableOp*^b_norm_2/AssignMovingAvg_1/ReadVariableOp^b_norm_2/ReadVariableOp^b_norm_2/ReadVariableOp_1-^b_norm_3/AssignMovingAvg/AssignSubVariableOp(^b_norm_3/AssignMovingAvg/ReadVariableOp/^b_norm_3/AssignMovingAvg_1/AssignSubVariableOp*^b_norm_3/AssignMovingAvg_1/ReadVariableOp^b_norm_3/ReadVariableOp^b_norm_3/ReadVariableOp_1-^b_norm_4/AssignMovingAvg/AssignSubVariableOp(^b_norm_4/AssignMovingAvg/ReadVariableOp/^b_norm_4/AssignMovingAvg_1/AssignSubVariableOp*^b_norm_4/AssignMovingAvg_1/ReadVariableOp^b_norm_4/ReadVariableOp^b_norm_4/ReadVariableOp_1^conv_1/BiasAdd/ReadVariableOp^conv_1/Conv2D/ReadVariableOp^conv_2/BiasAdd/ReadVariableOp^conv_2/Conv2D/ReadVariableOp^conv_3/BiasAdd/ReadVariableOp^conv_3/Conv2D/ReadVariableOp^conv_4/BiasAdd/ReadVariableOp^conv_4/Conv2D/ReadVariableOp>^input_batch_normalization/AssignMovingAvg/AssignSubVariableOp9^input_batch_normalization/AssignMovingAvg/ReadVariableOp@^input_batch_normalization/AssignMovingAvg_1/AssignSubVariableOp;^input_batch_normalization/AssignMovingAvg_1/ReadVariableOp)^input_batch_normalization/ReadVariableOp+^input_batch_normalization/ReadVariableOp_1"^input_conv/BiasAdd/ReadVariableOp!^input_conv/Conv2D/ReadVariableOp1^output_class_distribution/BiasAdd/ReadVariableOp0^output_class_distribution/MatMul/ReadVariableOp*
T0*(
_output_shapes
:����������2

Identity"
identityIdentity:output:0*�
_input_shapes�
�:�����������::::::::::::::::::::::::::::::::2\
,b_norm_1/AssignMovingAvg/AssignSubVariableOp,b_norm_1/AssignMovingAvg/AssignSubVariableOp2R
'b_norm_1/AssignMovingAvg/ReadVariableOp'b_norm_1/AssignMovingAvg/ReadVariableOp2`
.b_norm_1/AssignMovingAvg_1/AssignSubVariableOp.b_norm_1/AssignMovingAvg_1/AssignSubVariableOp2V
)b_norm_1/AssignMovingAvg_1/ReadVariableOp)b_norm_1/AssignMovingAvg_1/ReadVariableOp22
b_norm_1/ReadVariableOpb_norm_1/ReadVariableOp26
b_norm_1/ReadVariableOp_1b_norm_1/ReadVariableOp_12\
,b_norm_2/AssignMovingAvg/AssignSubVariableOp,b_norm_2/AssignMovingAvg/AssignSubVariableOp2R
'b_norm_2/AssignMovingAvg/ReadVariableOp'b_norm_2/AssignMovingAvg/ReadVariableOp2`
.b_norm_2/AssignMovingAvg_1/AssignSubVariableOp.b_norm_2/AssignMovingAvg_1/AssignSubVariableOp2V
)b_norm_2/AssignMovingAvg_1/ReadVariableOp)b_norm_2/AssignMovingAvg_1/ReadVariableOp22
b_norm_2/ReadVariableOpb_norm_2/ReadVariableOp26
b_norm_2/ReadVariableOp_1b_norm_2/ReadVariableOp_12\
,b_norm_3/AssignMovingAvg/AssignSubVariableOp,b_norm_3/AssignMovingAvg/AssignSubVariableOp2R
'b_norm_3/AssignMovingAvg/ReadVariableOp'b_norm_3/AssignMovingAvg/ReadVariableOp2`
.b_norm_3/AssignMovingAvg_1/AssignSubVariableOp.b_norm_3/AssignMovingAvg_1/AssignSubVariableOp2V
)b_norm_3/AssignMovingAvg_1/ReadVariableOp)b_norm_3/AssignMovingAvg_1/ReadVariableOp22
b_norm_3/ReadVariableOpb_norm_3/ReadVariableOp26
b_norm_3/ReadVariableOp_1b_norm_3/ReadVariableOp_12\
,b_norm_4/AssignMovingAvg/AssignSubVariableOp,b_norm_4/AssignMovingAvg/AssignSubVariableOp2R
'b_norm_4/AssignMovingAvg/ReadVariableOp'b_norm_4/AssignMovingAvg/ReadVariableOp2`
.b_norm_4/AssignMovingAvg_1/AssignSubVariableOp.b_norm_4/AssignMovingAvg_1/AssignSubVariableOp2V
)b_norm_4/AssignMovingAvg_1/ReadVariableOp)b_norm_4/AssignMovingAvg_1/ReadVariableOp22
b_norm_4/ReadVariableOpb_norm_4/ReadVariableOp26
b_norm_4/ReadVariableOp_1b_norm_4/ReadVariableOp_12>
conv_1/BiasAdd/ReadVariableOpconv_1/BiasAdd/ReadVariableOp2<
conv_1/Conv2D/ReadVariableOpconv_1/Conv2D/ReadVariableOp2>
conv_2/BiasAdd/ReadVariableOpconv_2/BiasAdd/ReadVariableOp2<
conv_2/Conv2D/ReadVariableOpconv_2/Conv2D/ReadVariableOp2>
conv_3/BiasAdd/ReadVariableOpconv_3/BiasAdd/ReadVariableOp2<
conv_3/Conv2D/ReadVariableOpconv_3/Conv2D/ReadVariableOp2>
conv_4/BiasAdd/ReadVariableOpconv_4/BiasAdd/ReadVariableOp2<
conv_4/Conv2D/ReadVariableOpconv_4/Conv2D/ReadVariableOp2~
=input_batch_normalization/AssignMovingAvg/AssignSubVariableOp=input_batch_normalization/AssignMovingAvg/AssignSubVariableOp2t
8input_batch_normalization/AssignMovingAvg/ReadVariableOp8input_batch_normalization/AssignMovingAvg/ReadVariableOp2�
?input_batch_normalization/AssignMovingAvg_1/AssignSubVariableOp?input_batch_normalization/AssignMovingAvg_1/AssignSubVariableOp2x
:input_batch_normalization/AssignMovingAvg_1/ReadVariableOp:input_batch_normalization/AssignMovingAvg_1/ReadVariableOp2T
(input_batch_normalization/ReadVariableOp(input_batch_normalization/ReadVariableOp2X
*input_batch_normalization/ReadVariableOp_1*input_batch_normalization/ReadVariableOp_12F
!input_conv/BiasAdd/ReadVariableOp!input_conv/BiasAdd/ReadVariableOp2D
 input_conv/Conv2D/ReadVariableOp input_conv/Conv2D/ReadVariableOp2d
0output_class_distribution/BiasAdd/ReadVariableOp0output_class_distribution/BiasAdd/ReadVariableOp2b
/output_class_distribution/MatMul/ReadVariableOp/output_class_distribution/MatMul/ReadVariableOp:& "
 
_user_specified_nameinputs
�
�
*__inference_b_norm_2_layer_call_fn_4354723

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4*
Tin	
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*A
_output_shapes/
-:+��������������������������� *-
config_proto

GPU

CPU2*0J 8*N
fIRG
E__inference_b_norm_2_layer_call_and_return_conditional_losses_43525052
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+��������������������������� 2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+��������������������������� ::::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs
�
H
,__inference_flatten_21_layer_call_fn_4355158

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*(
_output_shapes
:����������*-
config_proto

GPU

CPU2*0J 8*P
fKRI
G__inference_flatten_21_layer_call_and_return_conditional_losses_43533802
PartitionedCallm
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:����������2

Identity"
identityIdentity:output:0*.
_input_shapes
:���������@:& "
 
_user_specified_nameinputs
�$
�
E__inference_b_norm_3_layer_call_and_return_conditional_losses_4353218

inputs
readvariableop_resource
readvariableop_1_resource
assignmovingavg_4353203
assignmovingavg_1_4353210
identity��#AssignMovingAvg/AssignSubVariableOp�AssignMovingAvg/ReadVariableOp�%AssignMovingAvg_1/AssignSubVariableOp� AssignMovingAvg_1/ReadVariableOp�ReadVariableOp�ReadVariableOp_1^
LogicalAnd/xConst*
_output_shapes
: *
dtype0
*
value	B
 Z2
LogicalAnd/x^
LogicalAnd/yConst*
_output_shapes
: *
dtype0
*
value	B
 Z2
LogicalAnd/yl

LogicalAnd
LogicalAndLogicalAnd/x:output:0LogicalAnd/y:output:0*
_output_shapes
: 2

LogicalAndt
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1Q
ConstConst*
_output_shapes
: *
dtype0*
valueB 2
ConstU
Const_1Const*
_output_shapes
: *
dtype0*
valueB 2	
Const_1�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0Const:output:0Const_1:output:0*
T0*
U0*K
_output_shapes9
7:���������@:@:@:@:@:*
epsilon%o�:2
FusedBatchNormV3W
Const_2Const*
_output_shapes
: *
dtype0*
valueB
 *�p}?2	
Const_2�
AssignMovingAvg/sub/xConst**
_class 
loc:@AssignMovingAvg/4353203*
_output_shapes
: *
dtype0*
valueB
 *  �?2
AssignMovingAvg/sub/x�
AssignMovingAvg/subSubAssignMovingAvg/sub/x:output:0Const_2:output:0*
T0**
_class 
loc:@AssignMovingAvg/4353203*
_output_shapes
: 2
AssignMovingAvg/sub�
AssignMovingAvg/ReadVariableOpReadVariableOpassignmovingavg_4353203*
_output_shapes
:@*
dtype02 
AssignMovingAvg/ReadVariableOp�
AssignMovingAvg/sub_1Sub&AssignMovingAvg/ReadVariableOp:value:0FusedBatchNormV3:batch_mean:0*
T0**
_class 
loc:@AssignMovingAvg/4353203*
_output_shapes
:@2
AssignMovingAvg/sub_1�
AssignMovingAvg/mulMulAssignMovingAvg/sub_1:z:0AssignMovingAvg/sub:z:0*
T0**
_class 
loc:@AssignMovingAvg/4353203*
_output_shapes
:@2
AssignMovingAvg/mul�
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOpassignmovingavg_4353203AssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp**
_class 
loc:@AssignMovingAvg/4353203*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOp�
AssignMovingAvg_1/sub/xConst*,
_class"
 loc:@AssignMovingAvg_1/4353210*
_output_shapes
: *
dtype0*
valueB
 *  �?2
AssignMovingAvg_1/sub/x�
AssignMovingAvg_1/subSub AssignMovingAvg_1/sub/x:output:0Const_2:output:0*
T0*,
_class"
 loc:@AssignMovingAvg_1/4353210*
_output_shapes
: 2
AssignMovingAvg_1/sub�
 AssignMovingAvg_1/ReadVariableOpReadVariableOpassignmovingavg_1_4353210*
_output_shapes
:@*
dtype02"
 AssignMovingAvg_1/ReadVariableOp�
AssignMovingAvg_1/sub_1Sub(AssignMovingAvg_1/ReadVariableOp:value:0!FusedBatchNormV3:batch_variance:0*
T0*,
_class"
 loc:@AssignMovingAvg_1/4353210*
_output_shapes
:@2
AssignMovingAvg_1/sub_1�
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub_1:z:0AssignMovingAvg_1/sub:z:0*
T0*,
_class"
 loc:@AssignMovingAvg_1/4353210*
_output_shapes
:@2
AssignMovingAvg_1/mul�
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOpassignmovingavg_1_4353210AssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*,
_class"
 loc:@AssignMovingAvg_1/4353210*
_output_shapes
 *
dtype02'
%AssignMovingAvg_1/AssignSubVariableOp�
IdentityIdentityFusedBatchNormV3:y:0$^AssignMovingAvg/AssignSubVariableOp^AssignMovingAvg/ReadVariableOp&^AssignMovingAvg_1/AssignSubVariableOp!^AssignMovingAvg_1/ReadVariableOp^ReadVariableOp^ReadVariableOp_1*
T0*/
_output_shapes
:���������@2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:���������@::::2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp2D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp2 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:& "
 
_user_specified_nameinputs
�
�
E__inference_b_norm_2_layer_call_and_return_conditional_losses_4353144

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity��FusedBatchNormV3/ReadVariableOp�!FusedBatchNormV3/ReadVariableOp_1�ReadVariableOp�ReadVariableOp_1^
LogicalAnd/xConst*
_output_shapes
: *
dtype0
*
value	B
 Z 2
LogicalAnd/x^
LogicalAnd/yConst*
_output_shapes
: *
dtype0
*
value	B
 Z2
LogicalAnd/yl

LogicalAnd
LogicalAndLogicalAnd/x:output:0LogicalAnd/y:output:0*
_output_shapes
: 2

LogicalAndt
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype02
ReadVariableOp_1�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02!
FusedBatchNormV3/ReadVariableOp�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02#
!FusedBatchNormV3/ReadVariableOp_1�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:���������   : : : : :*
epsilon%o�:*
is_training( 2
FusedBatchNormV3S
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *�p}?2
Const�
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*/
_output_shapes
:���������   2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:���������   ::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:& "
 
_user_specified_nameinputs
�$
�
V__inference_input_batch_normalization_layer_call_and_return_conditional_losses_4352892

inputs
readvariableop_resource
readvariableop_1_resource
assignmovingavg_4352877
assignmovingavg_1_4352884
identity��#AssignMovingAvg/AssignSubVariableOp�AssignMovingAvg/ReadVariableOp�%AssignMovingAvg_1/AssignSubVariableOp� AssignMovingAvg_1/ReadVariableOp�ReadVariableOp�ReadVariableOp_1^
LogicalAnd/xConst*
_output_shapes
: *
dtype0
*
value	B
 Z2
LogicalAnd/x^
LogicalAnd/yConst*
_output_shapes
: *
dtype0
*
value	B
 Z2
LogicalAnd/yl

LogicalAnd
LogicalAndLogicalAnd/x:output:0LogicalAnd/y:output:0*
_output_shapes
: 2

LogicalAndt
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1Q
ConstConst*
_output_shapes
: *
dtype0*
valueB 2
ConstU
Const_1Const*
_output_shapes
: *
dtype0*
valueB 2	
Const_1�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0Const:output:0Const_1:output:0*
T0*
U0*M
_output_shapes;
9:�����������:::::*
epsilon%o�:2
FusedBatchNormV3W
Const_2Const*
_output_shapes
: *
dtype0*
valueB
 *�p}?2	
Const_2�
AssignMovingAvg/sub/xConst**
_class 
loc:@AssignMovingAvg/4352877*
_output_shapes
: *
dtype0*
valueB
 *  �?2
AssignMovingAvg/sub/x�
AssignMovingAvg/subSubAssignMovingAvg/sub/x:output:0Const_2:output:0*
T0**
_class 
loc:@AssignMovingAvg/4352877*
_output_shapes
: 2
AssignMovingAvg/sub�
AssignMovingAvg/ReadVariableOpReadVariableOpassignmovingavg_4352877*
_output_shapes
:*
dtype02 
AssignMovingAvg/ReadVariableOp�
AssignMovingAvg/sub_1Sub&AssignMovingAvg/ReadVariableOp:value:0FusedBatchNormV3:batch_mean:0*
T0**
_class 
loc:@AssignMovingAvg/4352877*
_output_shapes
:2
AssignMovingAvg/sub_1�
AssignMovingAvg/mulMulAssignMovingAvg/sub_1:z:0AssignMovingAvg/sub:z:0*
T0**
_class 
loc:@AssignMovingAvg/4352877*
_output_shapes
:2
AssignMovingAvg/mul�
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOpassignmovingavg_4352877AssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp**
_class 
loc:@AssignMovingAvg/4352877*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOp�
AssignMovingAvg_1/sub/xConst*,
_class"
 loc:@AssignMovingAvg_1/4352884*
_output_shapes
: *
dtype0*
valueB
 *  �?2
AssignMovingAvg_1/sub/x�
AssignMovingAvg_1/subSub AssignMovingAvg_1/sub/x:output:0Const_2:output:0*
T0*,
_class"
 loc:@AssignMovingAvg_1/4352884*
_output_shapes
: 2
AssignMovingAvg_1/sub�
 AssignMovingAvg_1/ReadVariableOpReadVariableOpassignmovingavg_1_4352884*
_output_shapes
:*
dtype02"
 AssignMovingAvg_1/ReadVariableOp�
AssignMovingAvg_1/sub_1Sub(AssignMovingAvg_1/ReadVariableOp:value:0!FusedBatchNormV3:batch_variance:0*
T0*,
_class"
 loc:@AssignMovingAvg_1/4352884*
_output_shapes
:2
AssignMovingAvg_1/sub_1�
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub_1:z:0AssignMovingAvg_1/sub:z:0*
T0*,
_class"
 loc:@AssignMovingAvg_1/4352884*
_output_shapes
:2
AssignMovingAvg_1/mul�
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOpassignmovingavg_1_4352884AssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*,
_class"
 loc:@AssignMovingAvg_1/4352884*
_output_shapes
 *
dtype02'
%AssignMovingAvg_1/AssignSubVariableOp�
IdentityIdentityFusedBatchNormV3:y:0$^AssignMovingAvg/AssignSubVariableOp^AssignMovingAvg/ReadVariableOp&^AssignMovingAvg_1/AssignSubVariableOp!^AssignMovingAvg_1/ReadVariableOp^ReadVariableOp^ReadVariableOp_1*
T0*1
_output_shapes
:�����������2

Identity"
identityIdentity:output:0*@
_input_shapes/
-:�����������::::2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp2D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp2 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:& "
 
_user_specified_nameinputs
�
f
G__inference_dropout_42_layer_call_and_return_conditional_losses_4352972

inputs
identity�a
dropout/rateConst*
_output_shapes
: *
dtype0*
valueB
 *33�>2
dropout/rateT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape}
dropout/random_uniform/minConst*
_output_shapes
: *
dtype0*
valueB
 *    2
dropout/random_uniform/min}
dropout/random_uniform/maxConst*
_output_shapes
: *
dtype0*
valueB
 *  �?2
dropout/random_uniform/max�
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*/
_output_shapes
:���������@@*
dtype02&
$dropout/random_uniform/RandomUniform�
dropout/random_uniform/subSub#dropout/random_uniform/max:output:0#dropout/random_uniform/min:output:0*
T0*
_output_shapes
: 2
dropout/random_uniform/sub�
dropout/random_uniform/mulMul-dropout/random_uniform/RandomUniform:output:0dropout/random_uniform/sub:z:0*
T0*/
_output_shapes
:���������@@2
dropout/random_uniform/mul�
dropout/random_uniformAdddropout/random_uniform/mul:z:0#dropout/random_uniform/min:output:0*
T0*/
_output_shapes
:���������@@2
dropout/random_uniformc
dropout/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?2
dropout/sub/xq
dropout/subSubdropout/sub/x:output:0dropout/rate:output:0*
T0*
_output_shapes
: 2
dropout/subk
dropout/truediv/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?2
dropout/truediv/x{
dropout/truedivRealDivdropout/truediv/x:output:0dropout/sub:z:0*
T0*
_output_shapes
: 2
dropout/truediv�
dropout/GreaterEqualGreaterEqualdropout/random_uniform:z:0dropout/rate:output:0*
T0*/
_output_shapes
:���������@@2
dropout/GreaterEqualx
dropout/mulMulinputsdropout/truediv:z:0*
T0*/
_output_shapes
:���������@@2
dropout/mul�
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:���������@@2
dropout/Cast�
dropout/mul_1Muldropout/mul:z:0dropout/Cast:y:0*
T0*/
_output_shapes
:���������@@2
dropout/mul_1m
IdentityIdentitydropout/mul_1:z:0*
T0*/
_output_shapes
:���������@@2

Identity"
identityIdentity:output:0*.
_input_shapes
:���������@@:& "
 
_user_specified_nameinputs
�
�
*__inference_b_norm_1_layer_call_fn_4354544

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4*
Tin	
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*/
_output_shapes
:���������@@*-
config_proto

GPU

CPU2*0J 8*N
fIRG
E__inference_b_norm_1_layer_call_and_return_conditional_losses_43530262
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:���������@@2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:���������@@::::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs
��
�%
 __inference__traced_save_4355483
file_prefix3
/savev2_input_conv_21_kernel_read_readvariableop1
-savev2_input_conv_21_bias_read_readvariableopA
=savev2_input_batch_normalization_21_gamma_read_readvariableop@
<savev2_input_batch_normalization_21_beta_read_readvariableopG
Csavev2_input_batch_normalization_21_moving_mean_read_readvariableopK
Gsavev2_input_batch_normalization_21_moving_variance_read_readvariableop/
+savev2_conv_1_21_kernel_read_readvariableop-
)savev2_conv_1_21_bias_read_readvariableop0
,savev2_b_norm_1_21_gamma_read_readvariableop/
+savev2_b_norm_1_21_beta_read_readvariableop6
2savev2_b_norm_1_21_moving_mean_read_readvariableop:
6savev2_b_norm_1_21_moving_variance_read_readvariableop/
+savev2_conv_2_21_kernel_read_readvariableop-
)savev2_conv_2_21_bias_read_readvariableop0
,savev2_b_norm_2_21_gamma_read_readvariableop/
+savev2_b_norm_2_21_beta_read_readvariableop6
2savev2_b_norm_2_21_moving_mean_read_readvariableop:
6savev2_b_norm_2_21_moving_variance_read_readvariableop/
+savev2_conv_3_13_kernel_read_readvariableop-
)savev2_conv_3_13_bias_read_readvariableop0
,savev2_b_norm_3_13_gamma_read_readvariableop/
+savev2_b_norm_3_13_beta_read_readvariableop6
2savev2_b_norm_3_13_moving_mean_read_readvariableop:
6savev2_b_norm_3_13_moving_variance_read_readvariableop.
*savev2_conv_4_5_kernel_read_readvariableop,
(savev2_conv_4_5_bias_read_readvariableop/
+savev2_b_norm_4_5_gamma_read_readvariableop.
*savev2_b_norm_4_5_beta_read_readvariableop5
1savev2_b_norm_4_5_moving_mean_read_readvariableop9
5savev2_b_norm_4_5_moving_variance_read_readvariableopB
>savev2_output_class_distribution_21_kernel_read_readvariableop@
<savev2_output_class_distribution_21_bias_read_readvariableop(
$savev2_adam_iter_read_readvariableop	*
&savev2_adam_beta_1_read_readvariableop*
&savev2_adam_beta_2_read_readvariableop)
%savev2_adam_decay_read_readvariableop1
-savev2_adam_learning_rate_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop:
6savev2_adam_input_conv_21_kernel_m_read_readvariableop8
4savev2_adam_input_conv_21_bias_m_read_readvariableopH
Dsavev2_adam_input_batch_normalization_21_gamma_m_read_readvariableopG
Csavev2_adam_input_batch_normalization_21_beta_m_read_readvariableop6
2savev2_adam_conv_1_21_kernel_m_read_readvariableop4
0savev2_adam_conv_1_21_bias_m_read_readvariableop7
3savev2_adam_b_norm_1_21_gamma_m_read_readvariableop6
2savev2_adam_b_norm_1_21_beta_m_read_readvariableop6
2savev2_adam_conv_2_21_kernel_m_read_readvariableop4
0savev2_adam_conv_2_21_bias_m_read_readvariableop7
3savev2_adam_b_norm_2_21_gamma_m_read_readvariableop6
2savev2_adam_b_norm_2_21_beta_m_read_readvariableop6
2savev2_adam_conv_3_13_kernel_m_read_readvariableop4
0savev2_adam_conv_3_13_bias_m_read_readvariableop7
3savev2_adam_b_norm_3_13_gamma_m_read_readvariableop6
2savev2_adam_b_norm_3_13_beta_m_read_readvariableop5
1savev2_adam_conv_4_5_kernel_m_read_readvariableop3
/savev2_adam_conv_4_5_bias_m_read_readvariableop6
2savev2_adam_b_norm_4_5_gamma_m_read_readvariableop5
1savev2_adam_b_norm_4_5_beta_m_read_readvariableopI
Esavev2_adam_output_class_distribution_21_kernel_m_read_readvariableopG
Csavev2_adam_output_class_distribution_21_bias_m_read_readvariableop:
6savev2_adam_input_conv_21_kernel_v_read_readvariableop8
4savev2_adam_input_conv_21_bias_v_read_readvariableopH
Dsavev2_adam_input_batch_normalization_21_gamma_v_read_readvariableopG
Csavev2_adam_input_batch_normalization_21_beta_v_read_readvariableop6
2savev2_adam_conv_1_21_kernel_v_read_readvariableop4
0savev2_adam_conv_1_21_bias_v_read_readvariableop7
3savev2_adam_b_norm_1_21_gamma_v_read_readvariableop6
2savev2_adam_b_norm_1_21_beta_v_read_readvariableop6
2savev2_adam_conv_2_21_kernel_v_read_readvariableop4
0savev2_adam_conv_2_21_bias_v_read_readvariableop7
3savev2_adam_b_norm_2_21_gamma_v_read_readvariableop6
2savev2_adam_b_norm_2_21_beta_v_read_readvariableop6
2savev2_adam_conv_3_13_kernel_v_read_readvariableop4
0savev2_adam_conv_3_13_bias_v_read_readvariableop7
3savev2_adam_b_norm_3_13_gamma_v_read_readvariableop6
2savev2_adam_b_norm_3_13_beta_v_read_readvariableop5
1savev2_adam_conv_4_5_kernel_v_read_readvariableop3
/savev2_adam_conv_4_5_bias_v_read_readvariableop6
2savev2_adam_b_norm_4_5_gamma_v_read_readvariableop5
1savev2_adam_b_norm_4_5_beta_v_read_readvariableopI
Esavev2_adam_output_class_distribution_21_kernel_v_read_readvariableopG
Csavev2_adam_output_class_distribution_21_bias_v_read_readvariableop
savev2_1_const

identity_1��MergeV2Checkpoints�SaveV2�SaveV2_1�
StringJoin/inputs_1Const"/device:CPU:0*
_output_shapes
: *
dtype0*<
value3B1 B+_temp_0bf83b01b65442a7a9670619ae16e6ce/part2
StringJoin/inputs_1�

StringJoin
StringJoinfile_prefixStringJoin/inputs_1:output:0"/device:CPU:0*
N*
_output_shapes
: 2

StringJoinZ

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :2

num_shards
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : 2
ShardedFilename/shard�
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilename�.
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:S*
dtype0*�-
value�-B�-SB6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-1/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-1/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-1/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-3/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-3/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-3/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-5/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-5/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-5/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-7/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-7/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-7/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-7/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-8/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-8/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-9/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-9/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-9/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-9/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-10/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-10/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-1/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-3/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-5/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-7/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-7/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-8/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-8/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-9/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-9/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-10/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-10/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-1/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-3/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-5/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-7/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-7/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-8/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-8/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-9/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-9/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-10/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-10/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE2
SaveV2/tensor_names�
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:S*
dtype0*�
value�B�SB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
SaveV2/shape_and_slices�#
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0/savev2_input_conv_21_kernel_read_readvariableop-savev2_input_conv_21_bias_read_readvariableop=savev2_input_batch_normalization_21_gamma_read_readvariableop<savev2_input_batch_normalization_21_beta_read_readvariableopCsavev2_input_batch_normalization_21_moving_mean_read_readvariableopGsavev2_input_batch_normalization_21_moving_variance_read_readvariableop+savev2_conv_1_21_kernel_read_readvariableop)savev2_conv_1_21_bias_read_readvariableop,savev2_b_norm_1_21_gamma_read_readvariableop+savev2_b_norm_1_21_beta_read_readvariableop2savev2_b_norm_1_21_moving_mean_read_readvariableop6savev2_b_norm_1_21_moving_variance_read_readvariableop+savev2_conv_2_21_kernel_read_readvariableop)savev2_conv_2_21_bias_read_readvariableop,savev2_b_norm_2_21_gamma_read_readvariableop+savev2_b_norm_2_21_beta_read_readvariableop2savev2_b_norm_2_21_moving_mean_read_readvariableop6savev2_b_norm_2_21_moving_variance_read_readvariableop+savev2_conv_3_13_kernel_read_readvariableop)savev2_conv_3_13_bias_read_readvariableop,savev2_b_norm_3_13_gamma_read_readvariableop+savev2_b_norm_3_13_beta_read_readvariableop2savev2_b_norm_3_13_moving_mean_read_readvariableop6savev2_b_norm_3_13_moving_variance_read_readvariableop*savev2_conv_4_5_kernel_read_readvariableop(savev2_conv_4_5_bias_read_readvariableop+savev2_b_norm_4_5_gamma_read_readvariableop*savev2_b_norm_4_5_beta_read_readvariableop1savev2_b_norm_4_5_moving_mean_read_readvariableop5savev2_b_norm_4_5_moving_variance_read_readvariableop>savev2_output_class_distribution_21_kernel_read_readvariableop<savev2_output_class_distribution_21_bias_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop6savev2_adam_input_conv_21_kernel_m_read_readvariableop4savev2_adam_input_conv_21_bias_m_read_readvariableopDsavev2_adam_input_batch_normalization_21_gamma_m_read_readvariableopCsavev2_adam_input_batch_normalization_21_beta_m_read_readvariableop2savev2_adam_conv_1_21_kernel_m_read_readvariableop0savev2_adam_conv_1_21_bias_m_read_readvariableop3savev2_adam_b_norm_1_21_gamma_m_read_readvariableop2savev2_adam_b_norm_1_21_beta_m_read_readvariableop2savev2_adam_conv_2_21_kernel_m_read_readvariableop0savev2_adam_conv_2_21_bias_m_read_readvariableop3savev2_adam_b_norm_2_21_gamma_m_read_readvariableop2savev2_adam_b_norm_2_21_beta_m_read_readvariableop2savev2_adam_conv_3_13_kernel_m_read_readvariableop0savev2_adam_conv_3_13_bias_m_read_readvariableop3savev2_adam_b_norm_3_13_gamma_m_read_readvariableop2savev2_adam_b_norm_3_13_beta_m_read_readvariableop1savev2_adam_conv_4_5_kernel_m_read_readvariableop/savev2_adam_conv_4_5_bias_m_read_readvariableop2savev2_adam_b_norm_4_5_gamma_m_read_readvariableop1savev2_adam_b_norm_4_5_beta_m_read_readvariableopEsavev2_adam_output_class_distribution_21_kernel_m_read_readvariableopCsavev2_adam_output_class_distribution_21_bias_m_read_readvariableop6savev2_adam_input_conv_21_kernel_v_read_readvariableop4savev2_adam_input_conv_21_bias_v_read_readvariableopDsavev2_adam_input_batch_normalization_21_gamma_v_read_readvariableopCsavev2_adam_input_batch_normalization_21_beta_v_read_readvariableop2savev2_adam_conv_1_21_kernel_v_read_readvariableop0savev2_adam_conv_1_21_bias_v_read_readvariableop3savev2_adam_b_norm_1_21_gamma_v_read_readvariableop2savev2_adam_b_norm_1_21_beta_v_read_readvariableop2savev2_adam_conv_2_21_kernel_v_read_readvariableop0savev2_adam_conv_2_21_bias_v_read_readvariableop3savev2_adam_b_norm_2_21_gamma_v_read_readvariableop2savev2_adam_b_norm_2_21_beta_v_read_readvariableop2savev2_adam_conv_3_13_kernel_v_read_readvariableop0savev2_adam_conv_3_13_bias_v_read_readvariableop3savev2_adam_b_norm_3_13_gamma_v_read_readvariableop2savev2_adam_b_norm_3_13_beta_v_read_readvariableop1savev2_adam_conv_4_5_kernel_v_read_readvariableop/savev2_adam_conv_4_5_bias_v_read_readvariableop2savev2_adam_b_norm_4_5_gamma_v_read_readvariableop1savev2_adam_b_norm_4_5_beta_v_read_readvariableopEsavev2_adam_output_class_distribution_21_kernel_v_read_readvariableopCsavev2_adam_output_class_distribution_21_bias_v_read_readvariableop"/device:CPU:0*
_output_shapes
 *a
dtypesW
U2S	2
SaveV2�
ShardedFilename_1/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B :2
ShardedFilename_1/shard�
ShardedFilename_1ShardedFilenameStringJoin:output:0 ShardedFilename_1/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilename_1�
SaveV2_1/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*1
value(B&B_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2_1/tensor_names�
SaveV2_1/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*
valueB
B 2
SaveV2_1/shape_and_slices�
SaveV2_1SaveV2ShardedFilename_1:filename:0SaveV2_1/tensor_names:output:0"SaveV2_1/shape_and_slices:output:0savev2_1_const^SaveV2"/device:CPU:0*
_output_shapes
 *
dtypes
22

SaveV2_1�
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0ShardedFilename_1:filename:0^SaveV2	^SaveV2_1"/device:CPU:0*
N*
T0*
_output_shapes
:2(
&MergeV2Checkpoints/checkpoint_prefixes�
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix	^SaveV2_1"/device:CPU:0*
_output_shapes
 2
MergeV2Checkpointsr
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: 2

Identity�

Identity_1IdentityIdentity:output:0^MergeV2Checkpoints^SaveV2	^SaveV2_1*
T0*
_output_shapes
: 2

Identity_1"!

identity_1Identity_1:output:0*�
_input_shapes�
�: ::::::::::::: : : : : : : @:@:@:@:@:@:@@:@:@:@:@:@:
��:�: : : : : : : ::::::::: : : : : @:@:@:@:@@:@:@:@:
��:�::::::::: : : : : @:@:@:@:@@:@:@:@:
��:�: 2(
MergeV2CheckpointsMergeV2Checkpoints2
SaveV2SaveV22
SaveV2_1SaveV2_1:+ '
%
_user_specified_namefile_prefix
�
�
E__inference_b_norm_3_layer_call_and_return_conditional_losses_4353240

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity��FusedBatchNormV3/ReadVariableOp�!FusedBatchNormV3/ReadVariableOp_1�ReadVariableOp�ReadVariableOp_1^
LogicalAnd/xConst*
_output_shapes
: *
dtype0
*
value	B
 Z 2
LogicalAnd/x^
LogicalAnd/yConst*
_output_shapes
: *
dtype0
*
value	B
 Z2
LogicalAnd/yl

LogicalAnd
LogicalAndLogicalAnd/x:output:0LogicalAnd/y:output:0*
_output_shapes
: 2

LogicalAndt
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOp�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:���������@:@:@:@:@:*
epsilon%o�:*
is_training( 2
FusedBatchNormV3S
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *�p}?2
Const�
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*/
_output_shapes
:���������@2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:���������@::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:& "
 
_user_specified_nameinputs
�
�
*__inference_b_norm_3_layer_call_fn_4354893

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4*
Tin	
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*A
_output_shapes/
-:+���������������������������@*-
config_proto

GPU

CPU2*0J 8*N
fIRG
E__inference_b_norm_3_layer_call_and_return_conditional_losses_43526692
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+���������������������������@2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+���������������������������@::::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs
�
�
E__inference_b_norm_4_layer_call_and_return_conditional_losses_4352833

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity��FusedBatchNormV3/ReadVariableOp�!FusedBatchNormV3/ReadVariableOp_1�ReadVariableOp�ReadVariableOp_1^
LogicalAnd/xConst*
_output_shapes
: *
dtype0
*
value	B
 Z 2
LogicalAnd/x^
LogicalAnd/yConst*
_output_shapes
: *
dtype0
*
value	B
 Z2
LogicalAnd/yl

LogicalAnd
LogicalAndLogicalAnd/x:output:0LogicalAnd/y:output:0*
_output_shapes
: 2

LogicalAndt
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOp�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+���������������������������@:@:@:@:@:*
epsilon%o�:*
is_training( 2
FusedBatchNormV3S
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *�p}?2
Const�
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*A
_output_shapes/
-:+���������������������������@2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+���������������������������@::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:& "
 
_user_specified_nameinputs
�
D
(__inference_RELU_3_layer_call_fn_4354977

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*/
_output_shapes
:���������@*-
config_proto

GPU

CPU2*0J 8*L
fGRE
C__inference_RELU_3_layer_call_and_return_conditional_losses_43532692
PartitionedCallt
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:���������@2

Identity"
identityIdentity:output:0*.
_input_shapes
:���������@:& "
 
_user_specified_nameinputs
�$
�
E__inference_b_norm_1_layer_call_and_return_conditional_losses_4354513

inputs
readvariableop_resource
readvariableop_1_resource
assignmovingavg_4354498
assignmovingavg_1_4354505
identity��#AssignMovingAvg/AssignSubVariableOp�AssignMovingAvg/ReadVariableOp�%AssignMovingAvg_1/AssignSubVariableOp� AssignMovingAvg_1/ReadVariableOp�ReadVariableOp�ReadVariableOp_1^
LogicalAnd/xConst*
_output_shapes
: *
dtype0
*
value	B
 Z2
LogicalAnd/x^
LogicalAnd/yConst*
_output_shapes
: *
dtype0
*
value	B
 Z2
LogicalAnd/yl

LogicalAnd
LogicalAndLogicalAnd/x:output:0LogicalAnd/y:output:0*
_output_shapes
: 2

LogicalAndt
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1Q
ConstConst*
_output_shapes
: *
dtype0*
valueB 2
ConstU
Const_1Const*
_output_shapes
: *
dtype0*
valueB 2	
Const_1�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0Const:output:0Const_1:output:0*
T0*
U0*K
_output_shapes9
7:���������@@:::::*
epsilon%o�:2
FusedBatchNormV3W
Const_2Const*
_output_shapes
: *
dtype0*
valueB
 *�p}?2	
Const_2�
AssignMovingAvg/sub/xConst**
_class 
loc:@AssignMovingAvg/4354498*
_output_shapes
: *
dtype0*
valueB
 *  �?2
AssignMovingAvg/sub/x�
AssignMovingAvg/subSubAssignMovingAvg/sub/x:output:0Const_2:output:0*
T0**
_class 
loc:@AssignMovingAvg/4354498*
_output_shapes
: 2
AssignMovingAvg/sub�
AssignMovingAvg/ReadVariableOpReadVariableOpassignmovingavg_4354498*
_output_shapes
:*
dtype02 
AssignMovingAvg/ReadVariableOp�
AssignMovingAvg/sub_1Sub&AssignMovingAvg/ReadVariableOp:value:0FusedBatchNormV3:batch_mean:0*
T0**
_class 
loc:@AssignMovingAvg/4354498*
_output_shapes
:2
AssignMovingAvg/sub_1�
AssignMovingAvg/mulMulAssignMovingAvg/sub_1:z:0AssignMovingAvg/sub:z:0*
T0**
_class 
loc:@AssignMovingAvg/4354498*
_output_shapes
:2
AssignMovingAvg/mul�
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOpassignmovingavg_4354498AssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp**
_class 
loc:@AssignMovingAvg/4354498*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOp�
AssignMovingAvg_1/sub/xConst*,
_class"
 loc:@AssignMovingAvg_1/4354505*
_output_shapes
: *
dtype0*
valueB
 *  �?2
AssignMovingAvg_1/sub/x�
AssignMovingAvg_1/subSub AssignMovingAvg_1/sub/x:output:0Const_2:output:0*
T0*,
_class"
 loc:@AssignMovingAvg_1/4354505*
_output_shapes
: 2
AssignMovingAvg_1/sub�
 AssignMovingAvg_1/ReadVariableOpReadVariableOpassignmovingavg_1_4354505*
_output_shapes
:*
dtype02"
 AssignMovingAvg_1/ReadVariableOp�
AssignMovingAvg_1/sub_1Sub(AssignMovingAvg_1/ReadVariableOp:value:0!FusedBatchNormV3:batch_variance:0*
T0*,
_class"
 loc:@AssignMovingAvg_1/4354505*
_output_shapes
:2
AssignMovingAvg_1/sub_1�
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub_1:z:0AssignMovingAvg_1/sub:z:0*
T0*,
_class"
 loc:@AssignMovingAvg_1/4354505*
_output_shapes
:2
AssignMovingAvg_1/mul�
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOpassignmovingavg_1_4354505AssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*,
_class"
 loc:@AssignMovingAvg_1/4354505*
_output_shapes
 *
dtype02'
%AssignMovingAvg_1/AssignSubVariableOp�
IdentityIdentityFusedBatchNormV3:y:0$^AssignMovingAvg/AssignSubVariableOp^AssignMovingAvg/ReadVariableOp&^AssignMovingAvg_1/AssignSubVariableOp!^AssignMovingAvg_1/ReadVariableOp^ReadVariableOp^ReadVariableOp_1*
T0*/
_output_shapes
:���������@@2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:���������@@::::2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp2D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp2 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:& "
 
_user_specified_nameinputs
�$
�
E__inference_b_norm_1_layer_call_and_return_conditional_losses_4354587

inputs
readvariableop_resource
readvariableop_1_resource
assignmovingavg_4354572
assignmovingavg_1_4354579
identity��#AssignMovingAvg/AssignSubVariableOp�AssignMovingAvg/ReadVariableOp�%AssignMovingAvg_1/AssignSubVariableOp� AssignMovingAvg_1/ReadVariableOp�ReadVariableOp�ReadVariableOp_1^
LogicalAnd/xConst*
_output_shapes
: *
dtype0
*
value	B
 Z2
LogicalAnd/x^
LogicalAnd/yConst*
_output_shapes
: *
dtype0
*
value	B
 Z2
LogicalAnd/yl

LogicalAnd
LogicalAndLogicalAnd/x:output:0LogicalAnd/y:output:0*
_output_shapes
: 2

LogicalAndt
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1Q
ConstConst*
_output_shapes
: *
dtype0*
valueB 2
ConstU
Const_1Const*
_output_shapes
: *
dtype0*
valueB 2	
Const_1�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0Const:output:0Const_1:output:0*
T0*
U0*]
_output_shapesK
I:+���������������������������:::::*
epsilon%o�:2
FusedBatchNormV3W
Const_2Const*
_output_shapes
: *
dtype0*
valueB
 *�p}?2	
Const_2�
AssignMovingAvg/sub/xConst**
_class 
loc:@AssignMovingAvg/4354572*
_output_shapes
: *
dtype0*
valueB
 *  �?2
AssignMovingAvg/sub/x�
AssignMovingAvg/subSubAssignMovingAvg/sub/x:output:0Const_2:output:0*
T0**
_class 
loc:@AssignMovingAvg/4354572*
_output_shapes
: 2
AssignMovingAvg/sub�
AssignMovingAvg/ReadVariableOpReadVariableOpassignmovingavg_4354572*
_output_shapes
:*
dtype02 
AssignMovingAvg/ReadVariableOp�
AssignMovingAvg/sub_1Sub&AssignMovingAvg/ReadVariableOp:value:0FusedBatchNormV3:batch_mean:0*
T0**
_class 
loc:@AssignMovingAvg/4354572*
_output_shapes
:2
AssignMovingAvg/sub_1�
AssignMovingAvg/mulMulAssignMovingAvg/sub_1:z:0AssignMovingAvg/sub:z:0*
T0**
_class 
loc:@AssignMovingAvg/4354572*
_output_shapes
:2
AssignMovingAvg/mul�
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOpassignmovingavg_4354572AssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp**
_class 
loc:@AssignMovingAvg/4354572*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOp�
AssignMovingAvg_1/sub/xConst*,
_class"
 loc:@AssignMovingAvg_1/4354579*
_output_shapes
: *
dtype0*
valueB
 *  �?2
AssignMovingAvg_1/sub/x�
AssignMovingAvg_1/subSub AssignMovingAvg_1/sub/x:output:0Const_2:output:0*
T0*,
_class"
 loc:@AssignMovingAvg_1/4354579*
_output_shapes
: 2
AssignMovingAvg_1/sub�
 AssignMovingAvg_1/ReadVariableOpReadVariableOpassignmovingavg_1_4354579*
_output_shapes
:*
dtype02"
 AssignMovingAvg_1/ReadVariableOp�
AssignMovingAvg_1/sub_1Sub(AssignMovingAvg_1/ReadVariableOp:value:0!FusedBatchNormV3:batch_variance:0*
T0*,
_class"
 loc:@AssignMovingAvg_1/4354579*
_output_shapes
:2
AssignMovingAvg_1/sub_1�
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub_1:z:0AssignMovingAvg_1/sub:z:0*
T0*,
_class"
 loc:@AssignMovingAvg_1/4354579*
_output_shapes
:2
AssignMovingAvg_1/mul�
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOpassignmovingavg_1_4354579AssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*,
_class"
 loc:@AssignMovingAvg_1/4354579*
_output_shapes
 *
dtype02'
%AssignMovingAvg_1/AssignSubVariableOp�
IdentityIdentityFusedBatchNormV3:y:0$^AssignMovingAvg/AssignSubVariableOp^AssignMovingAvg/ReadVariableOp&^AssignMovingAvg_1/AssignSubVariableOp!^AssignMovingAvg_1/ReadVariableOp^ReadVariableOp^ReadVariableOp_1*
T0*A
_output_shapes/
-:+���������������������������2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+���������������������������::::2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp2D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp2 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:& "
 
_user_specified_nameinputs
�
�
*__inference_b_norm_3_layer_call_fn_4354884

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4*
Tin	
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*A
_output_shapes/
-:+���������������������������@*-
config_proto

GPU

CPU2*0J 8*N
fIRG
E__inference_b_norm_3_layer_call_and_return_conditional_losses_43526382
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+���������������������������@2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+���������������������������@::::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs
�
_
C__inference_RELU_4_layer_call_and_return_conditional_losses_4355142

inputs
identityV
ReluReluinputs*
T0*/
_output_shapes
:���������@2
Relun
IdentityIdentityRelu:activations:0*
T0*/
_output_shapes
:���������@2

Identity"
identityIdentity:output:0*.
_input_shapes
:���������@:& "
 
_user_specified_nameinputs
�	
�
V__inference_output_class_distribution_layer_call_and_return_conditional_losses_4353436

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
��*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
MatMul�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2	
BiasAdd�
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*(
_output_shapes
:����������2

Identity"
identityIdentity:output:0*/
_input_shapes
:����������::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:& "
 
_user_specified_nameinputs
�$
�
V__inference_input_batch_normalization_layer_call_and_return_conditional_losses_4352146

inputs
readvariableop_resource
readvariableop_1_resource
assignmovingavg_4352131
assignmovingavg_1_4352138
identity��#AssignMovingAvg/AssignSubVariableOp�AssignMovingAvg/ReadVariableOp�%AssignMovingAvg_1/AssignSubVariableOp� AssignMovingAvg_1/ReadVariableOp�ReadVariableOp�ReadVariableOp_1^
LogicalAnd/xConst*
_output_shapes
: *
dtype0
*
value	B
 Z2
LogicalAnd/x^
LogicalAnd/yConst*
_output_shapes
: *
dtype0
*
value	B
 Z2
LogicalAnd/yl

LogicalAnd
LogicalAndLogicalAnd/x:output:0LogicalAnd/y:output:0*
_output_shapes
: 2

LogicalAndt
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1Q
ConstConst*
_output_shapes
: *
dtype0*
valueB 2
ConstU
Const_1Const*
_output_shapes
: *
dtype0*
valueB 2	
Const_1�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0Const:output:0Const_1:output:0*
T0*
U0*]
_output_shapesK
I:+���������������������������:::::*
epsilon%o�:2
FusedBatchNormV3W
Const_2Const*
_output_shapes
: *
dtype0*
valueB
 *�p}?2	
Const_2�
AssignMovingAvg/sub/xConst**
_class 
loc:@AssignMovingAvg/4352131*
_output_shapes
: *
dtype0*
valueB
 *  �?2
AssignMovingAvg/sub/x�
AssignMovingAvg/subSubAssignMovingAvg/sub/x:output:0Const_2:output:0*
T0**
_class 
loc:@AssignMovingAvg/4352131*
_output_shapes
: 2
AssignMovingAvg/sub�
AssignMovingAvg/ReadVariableOpReadVariableOpassignmovingavg_4352131*
_output_shapes
:*
dtype02 
AssignMovingAvg/ReadVariableOp�
AssignMovingAvg/sub_1Sub&AssignMovingAvg/ReadVariableOp:value:0FusedBatchNormV3:batch_mean:0*
T0**
_class 
loc:@AssignMovingAvg/4352131*
_output_shapes
:2
AssignMovingAvg/sub_1�
AssignMovingAvg/mulMulAssignMovingAvg/sub_1:z:0AssignMovingAvg/sub:z:0*
T0**
_class 
loc:@AssignMovingAvg/4352131*
_output_shapes
:2
AssignMovingAvg/mul�
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOpassignmovingavg_4352131AssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp**
_class 
loc:@AssignMovingAvg/4352131*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOp�
AssignMovingAvg_1/sub/xConst*,
_class"
 loc:@AssignMovingAvg_1/4352138*
_output_shapes
: *
dtype0*
valueB
 *  �?2
AssignMovingAvg_1/sub/x�
AssignMovingAvg_1/subSub AssignMovingAvg_1/sub/x:output:0Const_2:output:0*
T0*,
_class"
 loc:@AssignMovingAvg_1/4352138*
_output_shapes
: 2
AssignMovingAvg_1/sub�
 AssignMovingAvg_1/ReadVariableOpReadVariableOpassignmovingavg_1_4352138*
_output_shapes
:*
dtype02"
 AssignMovingAvg_1/ReadVariableOp�
AssignMovingAvg_1/sub_1Sub(AssignMovingAvg_1/ReadVariableOp:value:0!FusedBatchNormV3:batch_variance:0*
T0*,
_class"
 loc:@AssignMovingAvg_1/4352138*
_output_shapes
:2
AssignMovingAvg_1/sub_1�
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub_1:z:0AssignMovingAvg_1/sub:z:0*
T0*,
_class"
 loc:@AssignMovingAvg_1/4352138*
_output_shapes
:2
AssignMovingAvg_1/mul�
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOpassignmovingavg_1_4352138AssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*,
_class"
 loc:@AssignMovingAvg_1/4352138*
_output_shapes
 *
dtype02'
%AssignMovingAvg_1/AssignSubVariableOp�
IdentityIdentityFusedBatchNormV3:y:0$^AssignMovingAvg/AssignSubVariableOp^AssignMovingAvg/ReadVariableOp&^AssignMovingAvg_1/AssignSubVariableOp!^AssignMovingAvg_1/ReadVariableOp^ReadVariableOp^ReadVariableOp_1*
T0*A
_output_shapes/
-:+���������������������������2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+���������������������������::::2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp2D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp2 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:& "
 
_user_specified_nameinputs
�$
�
E__inference_b_norm_1_layer_call_and_return_conditional_losses_4352310

inputs
readvariableop_resource
readvariableop_1_resource
assignmovingavg_4352295
assignmovingavg_1_4352302
identity��#AssignMovingAvg/AssignSubVariableOp�AssignMovingAvg/ReadVariableOp�%AssignMovingAvg_1/AssignSubVariableOp� AssignMovingAvg_1/ReadVariableOp�ReadVariableOp�ReadVariableOp_1^
LogicalAnd/xConst*
_output_shapes
: *
dtype0
*
value	B
 Z2
LogicalAnd/x^
LogicalAnd/yConst*
_output_shapes
: *
dtype0
*
value	B
 Z2
LogicalAnd/yl

LogicalAnd
LogicalAndLogicalAnd/x:output:0LogicalAnd/y:output:0*
_output_shapes
: 2

LogicalAndt
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1Q
ConstConst*
_output_shapes
: *
dtype0*
valueB 2
ConstU
Const_1Const*
_output_shapes
: *
dtype0*
valueB 2	
Const_1�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0Const:output:0Const_1:output:0*
T0*
U0*]
_output_shapesK
I:+���������������������������:::::*
epsilon%o�:2
FusedBatchNormV3W
Const_2Const*
_output_shapes
: *
dtype0*
valueB
 *�p}?2	
Const_2�
AssignMovingAvg/sub/xConst**
_class 
loc:@AssignMovingAvg/4352295*
_output_shapes
: *
dtype0*
valueB
 *  �?2
AssignMovingAvg/sub/x�
AssignMovingAvg/subSubAssignMovingAvg/sub/x:output:0Const_2:output:0*
T0**
_class 
loc:@AssignMovingAvg/4352295*
_output_shapes
: 2
AssignMovingAvg/sub�
AssignMovingAvg/ReadVariableOpReadVariableOpassignmovingavg_4352295*
_output_shapes
:*
dtype02 
AssignMovingAvg/ReadVariableOp�
AssignMovingAvg/sub_1Sub&AssignMovingAvg/ReadVariableOp:value:0FusedBatchNormV3:batch_mean:0*
T0**
_class 
loc:@AssignMovingAvg/4352295*
_output_shapes
:2
AssignMovingAvg/sub_1�
AssignMovingAvg/mulMulAssignMovingAvg/sub_1:z:0AssignMovingAvg/sub:z:0*
T0**
_class 
loc:@AssignMovingAvg/4352295*
_output_shapes
:2
AssignMovingAvg/mul�
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOpassignmovingavg_4352295AssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp**
_class 
loc:@AssignMovingAvg/4352295*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOp�
AssignMovingAvg_1/sub/xConst*,
_class"
 loc:@AssignMovingAvg_1/4352302*
_output_shapes
: *
dtype0*
valueB
 *  �?2
AssignMovingAvg_1/sub/x�
AssignMovingAvg_1/subSub AssignMovingAvg_1/sub/x:output:0Const_2:output:0*
T0*,
_class"
 loc:@AssignMovingAvg_1/4352302*
_output_shapes
: 2
AssignMovingAvg_1/sub�
 AssignMovingAvg_1/ReadVariableOpReadVariableOpassignmovingavg_1_4352302*
_output_shapes
:*
dtype02"
 AssignMovingAvg_1/ReadVariableOp�
AssignMovingAvg_1/sub_1Sub(AssignMovingAvg_1/ReadVariableOp:value:0!FusedBatchNormV3:batch_variance:0*
T0*,
_class"
 loc:@AssignMovingAvg_1/4352302*
_output_shapes
:2
AssignMovingAvg_1/sub_1�
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub_1:z:0AssignMovingAvg_1/sub:z:0*
T0*,
_class"
 loc:@AssignMovingAvg_1/4352302*
_output_shapes
:2
AssignMovingAvg_1/mul�
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOpassignmovingavg_1_4352302AssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*,
_class"
 loc:@AssignMovingAvg_1/4352302*
_output_shapes
 *
dtype02'
%AssignMovingAvg_1/AssignSubVariableOp�
IdentityIdentityFusedBatchNormV3:y:0$^AssignMovingAvg/AssignSubVariableOp^AssignMovingAvg/ReadVariableOp&^AssignMovingAvg_1/AssignSubVariableOp!^AssignMovingAvg_1/ReadVariableOp^ReadVariableOp^ReadVariableOp_1*
T0*A
_output_shapes/
-:+���������������������������2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+���������������������������::::2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp2D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp2 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:& "
 
_user_specified_nameinputs
�$
�
E__inference_b_norm_2_layer_call_and_return_conditional_losses_4354683

inputs
readvariableop_resource
readvariableop_1_resource
assignmovingavg_4354668
assignmovingavg_1_4354675
identity��#AssignMovingAvg/AssignSubVariableOp�AssignMovingAvg/ReadVariableOp�%AssignMovingAvg_1/AssignSubVariableOp� AssignMovingAvg_1/ReadVariableOp�ReadVariableOp�ReadVariableOp_1^
LogicalAnd/xConst*
_output_shapes
: *
dtype0
*
value	B
 Z2
LogicalAnd/x^
LogicalAnd/yConst*
_output_shapes
: *
dtype0
*
value	B
 Z2
LogicalAnd/yl

LogicalAnd
LogicalAndLogicalAnd/x:output:0LogicalAnd/y:output:0*
_output_shapes
: 2

LogicalAndt
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype02
ReadVariableOp_1Q
ConstConst*
_output_shapes
: *
dtype0*
valueB 2
ConstU
Const_1Const*
_output_shapes
: *
dtype0*
valueB 2	
Const_1�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0Const:output:0Const_1:output:0*
T0*
U0*]
_output_shapesK
I:+��������������������������� : : : : :*
epsilon%o�:2
FusedBatchNormV3W
Const_2Const*
_output_shapes
: *
dtype0*
valueB
 *�p}?2	
Const_2�
AssignMovingAvg/sub/xConst**
_class 
loc:@AssignMovingAvg/4354668*
_output_shapes
: *
dtype0*
valueB
 *  �?2
AssignMovingAvg/sub/x�
AssignMovingAvg/subSubAssignMovingAvg/sub/x:output:0Const_2:output:0*
T0**
_class 
loc:@AssignMovingAvg/4354668*
_output_shapes
: 2
AssignMovingAvg/sub�
AssignMovingAvg/ReadVariableOpReadVariableOpassignmovingavg_4354668*
_output_shapes
: *
dtype02 
AssignMovingAvg/ReadVariableOp�
AssignMovingAvg/sub_1Sub&AssignMovingAvg/ReadVariableOp:value:0FusedBatchNormV3:batch_mean:0*
T0**
_class 
loc:@AssignMovingAvg/4354668*
_output_shapes
: 2
AssignMovingAvg/sub_1�
AssignMovingAvg/mulMulAssignMovingAvg/sub_1:z:0AssignMovingAvg/sub:z:0*
T0**
_class 
loc:@AssignMovingAvg/4354668*
_output_shapes
: 2
AssignMovingAvg/mul�
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOpassignmovingavg_4354668AssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp**
_class 
loc:@AssignMovingAvg/4354668*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOp�
AssignMovingAvg_1/sub/xConst*,
_class"
 loc:@AssignMovingAvg_1/4354675*
_output_shapes
: *
dtype0*
valueB
 *  �?2
AssignMovingAvg_1/sub/x�
AssignMovingAvg_1/subSub AssignMovingAvg_1/sub/x:output:0Const_2:output:0*
T0*,
_class"
 loc:@AssignMovingAvg_1/4354675*
_output_shapes
: 2
AssignMovingAvg_1/sub�
 AssignMovingAvg_1/ReadVariableOpReadVariableOpassignmovingavg_1_4354675*
_output_shapes
: *
dtype02"
 AssignMovingAvg_1/ReadVariableOp�
AssignMovingAvg_1/sub_1Sub(AssignMovingAvg_1/ReadVariableOp:value:0!FusedBatchNormV3:batch_variance:0*
T0*,
_class"
 loc:@AssignMovingAvg_1/4354675*
_output_shapes
: 2
AssignMovingAvg_1/sub_1�
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub_1:z:0AssignMovingAvg_1/sub:z:0*
T0*,
_class"
 loc:@AssignMovingAvg_1/4354675*
_output_shapes
: 2
AssignMovingAvg_1/mul�
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOpassignmovingavg_1_4354675AssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*,
_class"
 loc:@AssignMovingAvg_1/4354675*
_output_shapes
 *
dtype02'
%AssignMovingAvg_1/AssignSubVariableOp�
IdentityIdentityFusedBatchNormV3:y:0$^AssignMovingAvg/AssignSubVariableOp^AssignMovingAvg/ReadVariableOp&^AssignMovingAvg_1/AssignSubVariableOp!^AssignMovingAvg_1/ReadVariableOp^ReadVariableOp^ReadVariableOp_1*
T0*A
_output_shapes/
-:+��������������������������� 2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+��������������������������� ::::2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp2D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp2 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:& "
 
_user_specified_nameinputs
�r
�
G__inference_awsome_net_layer_call_and_return_conditional_losses_4353572

inputs-
)input_conv_statefulpartitionedcall_args_1-
)input_conv_statefulpartitionedcall_args_2<
8input_batch_normalization_statefulpartitionedcall_args_1<
8input_batch_normalization_statefulpartitionedcall_args_2<
8input_batch_normalization_statefulpartitionedcall_args_3<
8input_batch_normalization_statefulpartitionedcall_args_4)
%conv_1_statefulpartitionedcall_args_1)
%conv_1_statefulpartitionedcall_args_2+
'b_norm_1_statefulpartitionedcall_args_1+
'b_norm_1_statefulpartitionedcall_args_2+
'b_norm_1_statefulpartitionedcall_args_3+
'b_norm_1_statefulpartitionedcall_args_4)
%conv_2_statefulpartitionedcall_args_1)
%conv_2_statefulpartitionedcall_args_2+
'b_norm_2_statefulpartitionedcall_args_1+
'b_norm_2_statefulpartitionedcall_args_2+
'b_norm_2_statefulpartitionedcall_args_3+
'b_norm_2_statefulpartitionedcall_args_4)
%conv_3_statefulpartitionedcall_args_1)
%conv_3_statefulpartitionedcall_args_2+
'b_norm_3_statefulpartitionedcall_args_1+
'b_norm_3_statefulpartitionedcall_args_2+
'b_norm_3_statefulpartitionedcall_args_3+
'b_norm_3_statefulpartitionedcall_args_4)
%conv_4_statefulpartitionedcall_args_1)
%conv_4_statefulpartitionedcall_args_2+
'b_norm_4_statefulpartitionedcall_args_1+
'b_norm_4_statefulpartitionedcall_args_2+
'b_norm_4_statefulpartitionedcall_args_3+
'b_norm_4_statefulpartitionedcall_args_4<
8output_class_distribution_statefulpartitionedcall_args_1<
8output_class_distribution_statefulpartitionedcall_args_2
identity�� b_norm_1/StatefulPartitionedCall� b_norm_2/StatefulPartitionedCall� b_norm_3/StatefulPartitionedCall� b_norm_4/StatefulPartitionedCall�conv_1/StatefulPartitionedCall�conv_2/StatefulPartitionedCall�conv_3/StatefulPartitionedCall�conv_4/StatefulPartitionedCall�"dropout_42/StatefulPartitionedCall�"dropout_43/StatefulPartitionedCall�1input_batch_normalization/StatefulPartitionedCall�"input_conv/StatefulPartitionedCall�1output_class_distribution/StatefulPartitionedCall�
"input_conv/StatefulPartitionedCallStatefulPartitionedCallinputs)input_conv_statefulpartitionedcall_args_1)input_conv_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*1
_output_shapes
:�����������*-
config_proto

GPU

CPU2*0J 8*P
fKRI
G__inference_input_conv_layer_call_and_return_conditional_losses_43520442$
"input_conv/StatefulPartitionedCall�
1input_batch_normalization/StatefulPartitionedCallStatefulPartitionedCall+input_conv/StatefulPartitionedCall:output:08input_batch_normalization_statefulpartitionedcall_args_18input_batch_normalization_statefulpartitionedcall_args_28input_batch_normalization_statefulpartitionedcall_args_38input_batch_normalization_statefulpartitionedcall_args_4*
Tin	
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*1
_output_shapes
:�����������*-
config_proto

GPU

CPU2*0J 8*_
fZRX
V__inference_input_batch_normalization_layer_call_and_return_conditional_losses_435289223
1input_batch_normalization/StatefulPartitionedCall�
input_RELU/PartitionedCallPartitionedCall:input_batch_normalization/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*1
_output_shapes
:�����������*-
config_proto

GPU

CPU2*0J 8*P
fKRI
G__inference_input_RELU_layer_call_and_return_conditional_losses_43529432
input_RELU/PartitionedCall�
 max_pooling2d_85/PartitionedCallPartitionedCall#input_RELU/PartitionedCall:output:0*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*/
_output_shapes
:���������@@*-
config_proto

GPU

CPU2*0J 8*V
fQRO
M__inference_max_pooling2d_85_layer_call_and_return_conditional_losses_43521902"
 max_pooling2d_85/PartitionedCall�
"dropout_42/StatefulPartitionedCallStatefulPartitionedCall)max_pooling2d_85/PartitionedCall:output:0*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*/
_output_shapes
:���������@@*-
config_proto

GPU

CPU2*0J 8*P
fKRI
G__inference_dropout_42_layer_call_and_return_conditional_losses_43529722$
"dropout_42/StatefulPartitionedCall�
conv_1/StatefulPartitionedCallStatefulPartitionedCall+dropout_42/StatefulPartitionedCall:output:0%conv_1_statefulpartitionedcall_args_1%conv_1_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*/
_output_shapes
:���������@@*-
config_proto

GPU

CPU2*0J 8*L
fGRE
C__inference_conv_1_layer_call_and_return_conditional_losses_43522082 
conv_1/StatefulPartitionedCall�
 b_norm_1/StatefulPartitionedCallStatefulPartitionedCall'conv_1/StatefulPartitionedCall:output:0'b_norm_1_statefulpartitionedcall_args_1'b_norm_1_statefulpartitionedcall_args_2'b_norm_1_statefulpartitionedcall_args_3'b_norm_1_statefulpartitionedcall_args_4*
Tin	
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*/
_output_shapes
:���������@@*-
config_proto

GPU

CPU2*0J 8*N
fIRG
E__inference_b_norm_1_layer_call_and_return_conditional_losses_43530262"
 b_norm_1/StatefulPartitionedCall�
RELU_1/PartitionedCallPartitionedCall)b_norm_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*/
_output_shapes
:���������@@*-
config_proto

GPU

CPU2*0J 8*L
fGRE
C__inference_RELU_1_layer_call_and_return_conditional_losses_43530772
RELU_1/PartitionedCall�
 max_pooling2d_81/PartitionedCallPartitionedCallRELU_1/PartitionedCall:output:0*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*/
_output_shapes
:���������  *-
config_proto

GPU

CPU2*0J 8*V
fQRO
M__inference_max_pooling2d_81_layer_call_and_return_conditional_losses_43523542"
 max_pooling2d_81/PartitionedCall�
conv_2/StatefulPartitionedCallStatefulPartitionedCall)max_pooling2d_81/PartitionedCall:output:0%conv_2_statefulpartitionedcall_args_1%conv_2_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*/
_output_shapes
:���������   *-
config_proto

GPU

CPU2*0J 8*L
fGRE
C__inference_conv_2_layer_call_and_return_conditional_losses_43523722 
conv_2/StatefulPartitionedCall�
 b_norm_2/StatefulPartitionedCallStatefulPartitionedCall'conv_2/StatefulPartitionedCall:output:0'b_norm_2_statefulpartitionedcall_args_1'b_norm_2_statefulpartitionedcall_args_2'b_norm_2_statefulpartitionedcall_args_3'b_norm_2_statefulpartitionedcall_args_4*
Tin	
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*/
_output_shapes
:���������   *-
config_proto

GPU

CPU2*0J 8*N
fIRG
E__inference_b_norm_2_layer_call_and_return_conditional_losses_43531222"
 b_norm_2/StatefulPartitionedCall�
RELU_2/PartitionedCallPartitionedCall)b_norm_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*/
_output_shapes
:���������   *-
config_proto

GPU

CPU2*0J 8*L
fGRE
C__inference_RELU_2_layer_call_and_return_conditional_losses_43531732
RELU_2/PartitionedCall�
 max_pooling2d_82/PartitionedCallPartitionedCallRELU_2/PartitionedCall:output:0*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*/
_output_shapes
:��������� *-
config_proto

GPU

CPU2*0J 8*V
fQRO
M__inference_max_pooling2d_82_layer_call_and_return_conditional_losses_43525182"
 max_pooling2d_82/PartitionedCall�
conv_3/StatefulPartitionedCallStatefulPartitionedCall)max_pooling2d_82/PartitionedCall:output:0%conv_3_statefulpartitionedcall_args_1%conv_3_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*/
_output_shapes
:���������@*-
config_proto

GPU

CPU2*0J 8*L
fGRE
C__inference_conv_3_layer_call_and_return_conditional_losses_43525362 
conv_3/StatefulPartitionedCall�
 b_norm_3/StatefulPartitionedCallStatefulPartitionedCall'conv_3/StatefulPartitionedCall:output:0'b_norm_3_statefulpartitionedcall_args_1'b_norm_3_statefulpartitionedcall_args_2'b_norm_3_statefulpartitionedcall_args_3'b_norm_3_statefulpartitionedcall_args_4*
Tin	
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*/
_output_shapes
:���������@*-
config_proto

GPU

CPU2*0J 8*N
fIRG
E__inference_b_norm_3_layer_call_and_return_conditional_losses_43532182"
 b_norm_3/StatefulPartitionedCall�
RELU_3/PartitionedCallPartitionedCall)b_norm_3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*/
_output_shapes
:���������@*-
config_proto

GPU

CPU2*0J 8*L
fGRE
C__inference_RELU_3_layer_call_and_return_conditional_losses_43532692
RELU_3/PartitionedCall�
 max_pooling2d_83/PartitionedCallPartitionedCallRELU_3/PartitionedCall:output:0*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*/
_output_shapes
:���������@*-
config_proto

GPU

CPU2*0J 8*V
fQRO
M__inference_max_pooling2d_83_layer_call_and_return_conditional_losses_43526822"
 max_pooling2d_83/PartitionedCall�
conv_4/StatefulPartitionedCallStatefulPartitionedCall)max_pooling2d_83/PartitionedCall:output:0%conv_4_statefulpartitionedcall_args_1%conv_4_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*/
_output_shapes
:���������@*-
config_proto

GPU

CPU2*0J 8*L
fGRE
C__inference_conv_4_layer_call_and_return_conditional_losses_43527002 
conv_4/StatefulPartitionedCall�
 b_norm_4/StatefulPartitionedCallStatefulPartitionedCall'conv_4/StatefulPartitionedCall:output:0'b_norm_4_statefulpartitionedcall_args_1'b_norm_4_statefulpartitionedcall_args_2'b_norm_4_statefulpartitionedcall_args_3'b_norm_4_statefulpartitionedcall_args_4*
Tin	
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*/
_output_shapes
:���������@*-
config_proto

GPU

CPU2*0J 8*N
fIRG
E__inference_b_norm_4_layer_call_and_return_conditional_losses_43533142"
 b_norm_4/StatefulPartitionedCall�
RELU_4/PartitionedCallPartitionedCall)b_norm_4/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*/
_output_shapes
:���������@*-
config_proto

GPU

CPU2*0J 8*L
fGRE
C__inference_RELU_4_layer_call_and_return_conditional_losses_43533652
RELU_4/PartitionedCall�
 max_pooling2d_84/PartitionedCallPartitionedCallRELU_4/PartitionedCall:output:0*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*/
_output_shapes
:���������@*-
config_proto

GPU

CPU2*0J 8*V
fQRO
M__inference_max_pooling2d_84_layer_call_and_return_conditional_losses_43528462"
 max_pooling2d_84/PartitionedCall�
flatten_21/PartitionedCallPartitionedCall)max_pooling2d_84/PartitionedCall:output:0*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*(
_output_shapes
:����������*-
config_proto

GPU

CPU2*0J 8*P
fKRI
G__inference_flatten_21_layer_call_and_return_conditional_losses_43533802
flatten_21/PartitionedCall�
"dropout_43/StatefulPartitionedCallStatefulPartitionedCall#flatten_21/PartitionedCall:output:0#^dropout_42/StatefulPartitionedCall*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*(
_output_shapes
:����������*-
config_proto

GPU

CPU2*0J 8*P
fKRI
G__inference_dropout_43_layer_call_and_return_conditional_losses_43534082$
"dropout_43/StatefulPartitionedCall�
1output_class_distribution/StatefulPartitionedCallStatefulPartitionedCall+dropout_43/StatefulPartitionedCall:output:08output_class_distribution_statefulpartitionedcall_args_18output_class_distribution_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*(
_output_shapes
:����������*-
config_proto

GPU

CPU2*0J 8*_
fZRX
V__inference_output_class_distribution_layer_call_and_return_conditional_losses_435343623
1output_class_distribution/StatefulPartitionedCall�
IdentityIdentity:output_class_distribution/StatefulPartitionedCall:output:0!^b_norm_1/StatefulPartitionedCall!^b_norm_2/StatefulPartitionedCall!^b_norm_3/StatefulPartitionedCall!^b_norm_4/StatefulPartitionedCall^conv_1/StatefulPartitionedCall^conv_2/StatefulPartitionedCall^conv_3/StatefulPartitionedCall^conv_4/StatefulPartitionedCall#^dropout_42/StatefulPartitionedCall#^dropout_43/StatefulPartitionedCall2^input_batch_normalization/StatefulPartitionedCall#^input_conv/StatefulPartitionedCall2^output_class_distribution/StatefulPartitionedCall*
T0*(
_output_shapes
:����������2

Identity"
identityIdentity:output:0*�
_input_shapes�
�:�����������::::::::::::::::::::::::::::::::2D
 b_norm_1/StatefulPartitionedCall b_norm_1/StatefulPartitionedCall2D
 b_norm_2/StatefulPartitionedCall b_norm_2/StatefulPartitionedCall2D
 b_norm_3/StatefulPartitionedCall b_norm_3/StatefulPartitionedCall2D
 b_norm_4/StatefulPartitionedCall b_norm_4/StatefulPartitionedCall2@
conv_1/StatefulPartitionedCallconv_1/StatefulPartitionedCall2@
conv_2/StatefulPartitionedCallconv_2/StatefulPartitionedCall2@
conv_3/StatefulPartitionedCallconv_3/StatefulPartitionedCall2@
conv_4/StatefulPartitionedCallconv_4/StatefulPartitionedCall2H
"dropout_42/StatefulPartitionedCall"dropout_42/StatefulPartitionedCall2H
"dropout_43/StatefulPartitionedCall"dropout_43/StatefulPartitionedCall2f
1input_batch_normalization/StatefulPartitionedCall1input_batch_normalization/StatefulPartitionedCall2H
"input_conv/StatefulPartitionedCall"input_conv/StatefulPartitionedCall2f
1output_class_distribution/StatefulPartitionedCall1output_class_distribution/StatefulPartitionedCall:& "
 
_user_specified_nameinputs
�
_
C__inference_RELU_4_layer_call_and_return_conditional_losses_4353365

inputs
identityV
ReluReluinputs*
T0*/
_output_shapes
:���������@2
Relun
IdentityIdentityRelu:activations:0*
T0*/
_output_shapes
:���������@2

Identity"
identityIdentity:output:0*.
_input_shapes
:���������@:& "
 
_user_specified_nameinputs
�
H
,__inference_dropout_42_layer_call_fn_4354467

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*/
_output_shapes
:���������@@*-
config_proto

GPU

CPU2*0J 8*P
fKRI
G__inference_dropout_42_layer_call_and_return_conditional_losses_43529772
PartitionedCallt
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:���������@@2

Identity"
identityIdentity:output:0*.
_input_shapes
:���������@@:& "
 
_user_specified_nameinputs
�
�	
%__inference_signature_wrapper_4353810
input_conv_input"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4"
statefulpartitionedcall_args_5"
statefulpartitionedcall_args_6"
statefulpartitionedcall_args_7"
statefulpartitionedcall_args_8"
statefulpartitionedcall_args_9#
statefulpartitionedcall_args_10#
statefulpartitionedcall_args_11#
statefulpartitionedcall_args_12#
statefulpartitionedcall_args_13#
statefulpartitionedcall_args_14#
statefulpartitionedcall_args_15#
statefulpartitionedcall_args_16#
statefulpartitionedcall_args_17#
statefulpartitionedcall_args_18#
statefulpartitionedcall_args_19#
statefulpartitionedcall_args_20#
statefulpartitionedcall_args_21#
statefulpartitionedcall_args_22#
statefulpartitionedcall_args_23#
statefulpartitionedcall_args_24#
statefulpartitionedcall_args_25#
statefulpartitionedcall_args_26#
statefulpartitionedcall_args_27#
statefulpartitionedcall_args_28#
statefulpartitionedcall_args_29#
statefulpartitionedcall_args_30#
statefulpartitionedcall_args_31#
statefulpartitionedcall_args_32
identity��StatefulPartitionedCall�

StatefulPartitionedCallStatefulPartitionedCallinput_conv_inputstatefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4statefulpartitionedcall_args_5statefulpartitionedcall_args_6statefulpartitionedcall_args_7statefulpartitionedcall_args_8statefulpartitionedcall_args_9statefulpartitionedcall_args_10statefulpartitionedcall_args_11statefulpartitionedcall_args_12statefulpartitionedcall_args_13statefulpartitionedcall_args_14statefulpartitionedcall_args_15statefulpartitionedcall_args_16statefulpartitionedcall_args_17statefulpartitionedcall_args_18statefulpartitionedcall_args_19statefulpartitionedcall_args_20statefulpartitionedcall_args_21statefulpartitionedcall_args_22statefulpartitionedcall_args_23statefulpartitionedcall_args_24statefulpartitionedcall_args_25statefulpartitionedcall_args_26statefulpartitionedcall_args_27statefulpartitionedcall_args_28statefulpartitionedcall_args_29statefulpartitionedcall_args_30statefulpartitionedcall_args_31statefulpartitionedcall_args_32*,
Tin%
#2!*
Tout
2*,
_gradient_op_typePartitionedCallUnused*(
_output_shapes
:����������*-
config_proto

GPU

CPU2*0J 8*+
f&R$
"__inference__wrapped_model_43520322
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:����������2

Identity"
identityIdentity:output:0*�
_input_shapes�
�:�����������::::::::::::::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:0 ,
*
_user_specified_nameinput_conv_input
�
�	
,__inference_awsome_net_layer_call_fn_4354262

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4"
statefulpartitionedcall_args_5"
statefulpartitionedcall_args_6"
statefulpartitionedcall_args_7"
statefulpartitionedcall_args_8"
statefulpartitionedcall_args_9#
statefulpartitionedcall_args_10#
statefulpartitionedcall_args_11#
statefulpartitionedcall_args_12#
statefulpartitionedcall_args_13#
statefulpartitionedcall_args_14#
statefulpartitionedcall_args_15#
statefulpartitionedcall_args_16#
statefulpartitionedcall_args_17#
statefulpartitionedcall_args_18#
statefulpartitionedcall_args_19#
statefulpartitionedcall_args_20#
statefulpartitionedcall_args_21#
statefulpartitionedcall_args_22#
statefulpartitionedcall_args_23#
statefulpartitionedcall_args_24#
statefulpartitionedcall_args_25#
statefulpartitionedcall_args_26#
statefulpartitionedcall_args_27#
statefulpartitionedcall_args_28#
statefulpartitionedcall_args_29#
statefulpartitionedcall_args_30#
statefulpartitionedcall_args_31#
statefulpartitionedcall_args_32
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4statefulpartitionedcall_args_5statefulpartitionedcall_args_6statefulpartitionedcall_args_7statefulpartitionedcall_args_8statefulpartitionedcall_args_9statefulpartitionedcall_args_10statefulpartitionedcall_args_11statefulpartitionedcall_args_12statefulpartitionedcall_args_13statefulpartitionedcall_args_14statefulpartitionedcall_args_15statefulpartitionedcall_args_16statefulpartitionedcall_args_17statefulpartitionedcall_args_18statefulpartitionedcall_args_19statefulpartitionedcall_args_20statefulpartitionedcall_args_21statefulpartitionedcall_args_22statefulpartitionedcall_args_23statefulpartitionedcall_args_24statefulpartitionedcall_args_25statefulpartitionedcall_args_26statefulpartitionedcall_args_27statefulpartitionedcall_args_28statefulpartitionedcall_args_29statefulpartitionedcall_args_30statefulpartitionedcall_args_31statefulpartitionedcall_args_32*,
Tin%
#2!*
Tout
2*,
_gradient_op_typePartitionedCallUnused*(
_output_shapes
:����������*-
config_proto

GPU

CPU2*0J 8*P
fKRI
G__inference_awsome_net_layer_call_and_return_conditional_losses_43536692
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:����������2

Identity"
identityIdentity:output:0*�
_input_shapes�
�:�����������::::::::::::::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs
�
�
*__inference_b_norm_1_layer_call_fn_4354618

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4*
Tin	
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*A
_output_shapes/
-:+���������������������������*-
config_proto

GPU

CPU2*0J 8*N
fIRG
E__inference_b_norm_1_layer_call_and_return_conditional_losses_43523102
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+���������������������������2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+���������������������������::::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs
�
�
E__inference_b_norm_1_layer_call_and_return_conditional_losses_4352341

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity��FusedBatchNormV3/ReadVariableOp�!FusedBatchNormV3/ReadVariableOp_1�ReadVariableOp�ReadVariableOp_1^
LogicalAnd/xConst*
_output_shapes
: *
dtype0
*
value	B
 Z 2
LogicalAnd/x^
LogicalAnd/yConst*
_output_shapes
: *
dtype0
*
value	B
 Z2
LogicalAnd/yl

LogicalAnd
LogicalAndLogicalAnd/x:output:0LogicalAnd/y:output:0*
_output_shapes
: 2

LogicalAndt
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOp�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+���������������������������:::::*
epsilon%o�:*
is_training( 2
FusedBatchNormV3S
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *�p}?2
Const�
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*A
_output_shapes/
-:+���������������������������2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+���������������������������::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:& "
 
_user_specified_nameinputs
�$
�
E__inference_b_norm_4_layer_call_and_return_conditional_losses_4353314

inputs
readvariableop_resource
readvariableop_1_resource
assignmovingavg_4353299
assignmovingavg_1_4353306
identity��#AssignMovingAvg/AssignSubVariableOp�AssignMovingAvg/ReadVariableOp�%AssignMovingAvg_1/AssignSubVariableOp� AssignMovingAvg_1/ReadVariableOp�ReadVariableOp�ReadVariableOp_1^
LogicalAnd/xConst*
_output_shapes
: *
dtype0
*
value	B
 Z2
LogicalAnd/x^
LogicalAnd/yConst*
_output_shapes
: *
dtype0
*
value	B
 Z2
LogicalAnd/yl

LogicalAnd
LogicalAndLogicalAnd/x:output:0LogicalAnd/y:output:0*
_output_shapes
: 2

LogicalAndt
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1Q
ConstConst*
_output_shapes
: *
dtype0*
valueB 2
ConstU
Const_1Const*
_output_shapes
: *
dtype0*
valueB 2	
Const_1�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0Const:output:0Const_1:output:0*
T0*
U0*K
_output_shapes9
7:���������@:@:@:@:@:*
epsilon%o�:2
FusedBatchNormV3W
Const_2Const*
_output_shapes
: *
dtype0*
valueB
 *�p}?2	
Const_2�
AssignMovingAvg/sub/xConst**
_class 
loc:@AssignMovingAvg/4353299*
_output_shapes
: *
dtype0*
valueB
 *  �?2
AssignMovingAvg/sub/x�
AssignMovingAvg/subSubAssignMovingAvg/sub/x:output:0Const_2:output:0*
T0**
_class 
loc:@AssignMovingAvg/4353299*
_output_shapes
: 2
AssignMovingAvg/sub�
AssignMovingAvg/ReadVariableOpReadVariableOpassignmovingavg_4353299*
_output_shapes
:@*
dtype02 
AssignMovingAvg/ReadVariableOp�
AssignMovingAvg/sub_1Sub&AssignMovingAvg/ReadVariableOp:value:0FusedBatchNormV3:batch_mean:0*
T0**
_class 
loc:@AssignMovingAvg/4353299*
_output_shapes
:@2
AssignMovingAvg/sub_1�
AssignMovingAvg/mulMulAssignMovingAvg/sub_1:z:0AssignMovingAvg/sub:z:0*
T0**
_class 
loc:@AssignMovingAvg/4353299*
_output_shapes
:@2
AssignMovingAvg/mul�
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOpassignmovingavg_4353299AssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp**
_class 
loc:@AssignMovingAvg/4353299*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOp�
AssignMovingAvg_1/sub/xConst*,
_class"
 loc:@AssignMovingAvg_1/4353306*
_output_shapes
: *
dtype0*
valueB
 *  �?2
AssignMovingAvg_1/sub/x�
AssignMovingAvg_1/subSub AssignMovingAvg_1/sub/x:output:0Const_2:output:0*
T0*,
_class"
 loc:@AssignMovingAvg_1/4353306*
_output_shapes
: 2
AssignMovingAvg_1/sub�
 AssignMovingAvg_1/ReadVariableOpReadVariableOpassignmovingavg_1_4353306*
_output_shapes
:@*
dtype02"
 AssignMovingAvg_1/ReadVariableOp�
AssignMovingAvg_1/sub_1Sub(AssignMovingAvg_1/ReadVariableOp:value:0!FusedBatchNormV3:batch_variance:0*
T0*,
_class"
 loc:@AssignMovingAvg_1/4353306*
_output_shapes
:@2
AssignMovingAvg_1/sub_1�
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub_1:z:0AssignMovingAvg_1/sub:z:0*
T0*,
_class"
 loc:@AssignMovingAvg_1/4353306*
_output_shapes
:@2
AssignMovingAvg_1/mul�
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOpassignmovingavg_1_4353306AssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*,
_class"
 loc:@AssignMovingAvg_1/4353306*
_output_shapes
 *
dtype02'
%AssignMovingAvg_1/AssignSubVariableOp�
IdentityIdentityFusedBatchNormV3:y:0$^AssignMovingAvg/AssignSubVariableOp^AssignMovingAvg/ReadVariableOp&^AssignMovingAvg_1/AssignSubVariableOp!^AssignMovingAvg_1/ReadVariableOp^ReadVariableOp^ReadVariableOp_1*
T0*/
_output_shapes
:���������@2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:���������@::::2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp2D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp2 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:& "
 
_user_specified_nameinputs
�
�
*__inference_b_norm_4_layer_call_fn_4355137

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4*
Tin	
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*/
_output_shapes
:���������@*-
config_proto

GPU

CPU2*0J 8*N
fIRG
E__inference_b_norm_4_layer_call_and_return_conditional_losses_43533362
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:���������@2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:���������@::::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs
�
�
*__inference_b_norm_3_layer_call_fn_4354958

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4*
Tin	
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*/
_output_shapes
:���������@*-
config_proto

GPU

CPU2*0J 8*N
fIRG
E__inference_b_norm_3_layer_call_and_return_conditional_losses_43532182
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:���������@2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:���������@::::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs
�
_
C__inference_RELU_1_layer_call_and_return_conditional_losses_4353077

inputs
identityV
ReluReluinputs*
T0*/
_output_shapes
:���������@@2
Relun
IdentityIdentityRelu:activations:0*
T0*/
_output_shapes
:���������@@2

Identity"
identityIdentity:output:0*.
_input_shapes
:���������@@:& "
 
_user_specified_nameinputs
�
c
G__inference_input_RELU_layer_call_and_return_conditional_losses_4352943

inputs
identityX
ReluReluinputs*
T0*1
_output_shapes
:�����������2
Relup
IdentityIdentityRelu:activations:0*
T0*1
_output_shapes
:�����������2

Identity"
identityIdentity:output:0*0
_input_shapes
:�����������:& "
 
_user_specified_nameinputs
�
f
G__inference_dropout_43_layer_call_and_return_conditional_losses_4355178

inputs
identity�a
dropout/rateConst*
_output_shapes
: *
dtype0*
valueB
 *33�>2
dropout/rateT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape}
dropout/random_uniform/minConst*
_output_shapes
: *
dtype0*
valueB
 *    2
dropout/random_uniform/min}
dropout/random_uniform/maxConst*
_output_shapes
: *
dtype0*
valueB
 *  �?2
dropout/random_uniform/max�
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*(
_output_shapes
:����������*
dtype02&
$dropout/random_uniform/RandomUniform�
dropout/random_uniform/subSub#dropout/random_uniform/max:output:0#dropout/random_uniform/min:output:0*
T0*
_output_shapes
: 2
dropout/random_uniform/sub�
dropout/random_uniform/mulMul-dropout/random_uniform/RandomUniform:output:0dropout/random_uniform/sub:z:0*
T0*(
_output_shapes
:����������2
dropout/random_uniform/mul�
dropout/random_uniformAdddropout/random_uniform/mul:z:0#dropout/random_uniform/min:output:0*
T0*(
_output_shapes
:����������2
dropout/random_uniformc
dropout/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?2
dropout/sub/xq
dropout/subSubdropout/sub/x:output:0dropout/rate:output:0*
T0*
_output_shapes
: 2
dropout/subk
dropout/truediv/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?2
dropout/truediv/x{
dropout/truedivRealDivdropout/truediv/x:output:0dropout/sub:z:0*
T0*
_output_shapes
: 2
dropout/truediv�
dropout/GreaterEqualGreaterEqualdropout/random_uniform:z:0dropout/rate:output:0*
T0*(
_output_shapes
:����������2
dropout/GreaterEqualq
dropout/mulMulinputsdropout/truediv:z:0*
T0*(
_output_shapes
:����������2
dropout/mul�
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:����������2
dropout/Cast{
dropout/mul_1Muldropout/mul:z:0dropout/Cast:y:0*
T0*(
_output_shapes
:����������2
dropout/mul_1f
IdentityIdentitydropout/mul_1:z:0*
T0*(
_output_shapes
:����������2

Identity"
identityIdentity:output:0*'
_input_shapes
:����������:& "
 
_user_specified_nameinputs
�
i
M__inference_max_pooling2d_83_layer_call_and_return_conditional_losses_4352682

inputs
identity�
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4������������������������������������*
ksize
*
paddingVALID*
strides
2	
MaxPool�
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4������������������������������������2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4������������������������������������:& "
 
_user_specified_nameinputs
�
c
G__inference_input_RELU_layer_call_and_return_conditional_losses_4354427

inputs
identityX
ReluReluinputs*
T0*1
_output_shapes
:�����������2
Relup
IdentityIdentityRelu:activations:0*
T0*1
_output_shapes
:�����������2

Identity"
identityIdentity:output:0*0
_input_shapes
:�����������:& "
 
_user_specified_nameinputs
�
_
C__inference_RELU_3_layer_call_and_return_conditional_losses_4353269

inputs
identityV
ReluReluinputs*
T0*/
_output_shapes
:���������@2
Relun
IdentityIdentityRelu:activations:0*
T0*/
_output_shapes
:���������@2

Identity"
identityIdentity:output:0*.
_input_shapes
:���������@:& "
 
_user_specified_nameinputs
�
N
2__inference_max_pooling2d_82_layer_call_fn_4352524

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*J
_output_shapes8
6:4������������������������������������*-
config_proto

GPU

CPU2*0J 8*V
fQRO
M__inference_max_pooling2d_82_layer_call_and_return_conditional_losses_43525182
PartitionedCall�
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4������������������������������������2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4������������������������������������:& "
 
_user_specified_nameinputs
�
�
*__inference_b_norm_1_layer_call_fn_4354553

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4*
Tin	
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*/
_output_shapes
:���������@@*-
config_proto

GPU

CPU2*0J 8*N
fIRG
E__inference_b_norm_1_layer_call_and_return_conditional_losses_43530482
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:���������@@2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:���������@@::::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs
�$
�
E__inference_b_norm_3_layer_call_and_return_conditional_losses_4354853

inputs
readvariableop_resource
readvariableop_1_resource
assignmovingavg_4354838
assignmovingavg_1_4354845
identity��#AssignMovingAvg/AssignSubVariableOp�AssignMovingAvg/ReadVariableOp�%AssignMovingAvg_1/AssignSubVariableOp� AssignMovingAvg_1/ReadVariableOp�ReadVariableOp�ReadVariableOp_1^
LogicalAnd/xConst*
_output_shapes
: *
dtype0
*
value	B
 Z2
LogicalAnd/x^
LogicalAnd/yConst*
_output_shapes
: *
dtype0
*
value	B
 Z2
LogicalAnd/yl

LogicalAnd
LogicalAndLogicalAnd/x:output:0LogicalAnd/y:output:0*
_output_shapes
: 2

LogicalAndt
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1Q
ConstConst*
_output_shapes
: *
dtype0*
valueB 2
ConstU
Const_1Const*
_output_shapes
: *
dtype0*
valueB 2	
Const_1�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0Const:output:0Const_1:output:0*
T0*
U0*]
_output_shapesK
I:+���������������������������@:@:@:@:@:*
epsilon%o�:2
FusedBatchNormV3W
Const_2Const*
_output_shapes
: *
dtype0*
valueB
 *�p}?2	
Const_2�
AssignMovingAvg/sub/xConst**
_class 
loc:@AssignMovingAvg/4354838*
_output_shapes
: *
dtype0*
valueB
 *  �?2
AssignMovingAvg/sub/x�
AssignMovingAvg/subSubAssignMovingAvg/sub/x:output:0Const_2:output:0*
T0**
_class 
loc:@AssignMovingAvg/4354838*
_output_shapes
: 2
AssignMovingAvg/sub�
AssignMovingAvg/ReadVariableOpReadVariableOpassignmovingavg_4354838*
_output_shapes
:@*
dtype02 
AssignMovingAvg/ReadVariableOp�
AssignMovingAvg/sub_1Sub&AssignMovingAvg/ReadVariableOp:value:0FusedBatchNormV3:batch_mean:0*
T0**
_class 
loc:@AssignMovingAvg/4354838*
_output_shapes
:@2
AssignMovingAvg/sub_1�
AssignMovingAvg/mulMulAssignMovingAvg/sub_1:z:0AssignMovingAvg/sub:z:0*
T0**
_class 
loc:@AssignMovingAvg/4354838*
_output_shapes
:@2
AssignMovingAvg/mul�
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOpassignmovingavg_4354838AssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp**
_class 
loc:@AssignMovingAvg/4354838*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOp�
AssignMovingAvg_1/sub/xConst*,
_class"
 loc:@AssignMovingAvg_1/4354845*
_output_shapes
: *
dtype0*
valueB
 *  �?2
AssignMovingAvg_1/sub/x�
AssignMovingAvg_1/subSub AssignMovingAvg_1/sub/x:output:0Const_2:output:0*
T0*,
_class"
 loc:@AssignMovingAvg_1/4354845*
_output_shapes
: 2
AssignMovingAvg_1/sub�
 AssignMovingAvg_1/ReadVariableOpReadVariableOpassignmovingavg_1_4354845*
_output_shapes
:@*
dtype02"
 AssignMovingAvg_1/ReadVariableOp�
AssignMovingAvg_1/sub_1Sub(AssignMovingAvg_1/ReadVariableOp:value:0!FusedBatchNormV3:batch_variance:0*
T0*,
_class"
 loc:@AssignMovingAvg_1/4354845*
_output_shapes
:@2
AssignMovingAvg_1/sub_1�
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub_1:z:0AssignMovingAvg_1/sub:z:0*
T0*,
_class"
 loc:@AssignMovingAvg_1/4354845*
_output_shapes
:@2
AssignMovingAvg_1/mul�
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOpassignmovingavg_1_4354845AssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*,
_class"
 loc:@AssignMovingAvg_1/4354845*
_output_shapes
 *
dtype02'
%AssignMovingAvg_1/AssignSubVariableOp�
IdentityIdentityFusedBatchNormV3:y:0$^AssignMovingAvg/AssignSubVariableOp^AssignMovingAvg/ReadVariableOp&^AssignMovingAvg_1/AssignSubVariableOp!^AssignMovingAvg_1/ReadVariableOp^ReadVariableOp^ReadVariableOp_1*
T0*A
_output_shapes/
-:+���������������������������@2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+���������������������������@::::2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp2D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp2 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:& "
 
_user_specified_nameinputs
�

�
C__inference_conv_1_layer_call_and_return_conditional_losses_4352208

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOpo
dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      2
dilation_rate�
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype02
Conv2D/ReadVariableOp�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+���������������������������*
paddingSAME*
strides
2
Conv2D�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+���������������������������2	
BiasAdd�
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*A
_output_shapes/
-:+���������������������������2

Identity"
identityIdentity:output:0*H
_input_shapes7
5:+���������������������������::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:& "
 
_user_specified_nameinputs
�
�
V__inference_input_batch_normalization_layer_call_and_return_conditional_losses_4352914

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity��FusedBatchNormV3/ReadVariableOp�!FusedBatchNormV3/ReadVariableOp_1�ReadVariableOp�ReadVariableOp_1^
LogicalAnd/xConst*
_output_shapes
: *
dtype0
*
value	B
 Z 2
LogicalAnd/x^
LogicalAnd/yConst*
_output_shapes
: *
dtype0
*
value	B
 Z2
LogicalAnd/yl

LogicalAnd
LogicalAndLogicalAnd/x:output:0LogicalAnd/y:output:0*
_output_shapes
: 2

LogicalAndt
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOp�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*M
_output_shapes;
9:�����������:::::*
epsilon%o�:*
is_training( 2
FusedBatchNormV3S
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *�p}?2
Const�
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*1
_output_shapes
:�����������2

Identity"
identityIdentity:output:0*@
_input_shapes/
-:�����������::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:& "
 
_user_specified_nameinputs
�
�
E__inference_b_norm_3_layer_call_and_return_conditional_losses_4352669

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity��FusedBatchNormV3/ReadVariableOp�!FusedBatchNormV3/ReadVariableOp_1�ReadVariableOp�ReadVariableOp_1^
LogicalAnd/xConst*
_output_shapes
: *
dtype0
*
value	B
 Z 2
LogicalAnd/x^
LogicalAnd/yConst*
_output_shapes
: *
dtype0
*
value	B
 Z2
LogicalAnd/yl

LogicalAnd
LogicalAndLogicalAnd/x:output:0LogicalAnd/y:output:0*
_output_shapes
: 2

LogicalAndt
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOp�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+���������������������������@:@:@:@:@:*
epsilon%o�:*
is_training( 2
FusedBatchNormV3S
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *�p}?2
Const�
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*A
_output_shapes/
-:+���������������������������@2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+���������������������������@::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:& "
 
_user_specified_nameinputs
�
D
(__inference_RELU_4_layer_call_fn_4355147

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*/
_output_shapes
:���������@*-
config_proto

GPU

CPU2*0J 8*L
fGRE
C__inference_RELU_4_layer_call_and_return_conditional_losses_43533652
PartitionedCallt
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:���������@2

Identity"
identityIdentity:output:0*.
_input_shapes
:���������@:& "
 
_user_specified_nameinputs
�s
�
G__inference_awsome_net_layer_call_and_return_conditional_losses_4353449
input_conv_input-
)input_conv_statefulpartitionedcall_args_1-
)input_conv_statefulpartitionedcall_args_2<
8input_batch_normalization_statefulpartitionedcall_args_1<
8input_batch_normalization_statefulpartitionedcall_args_2<
8input_batch_normalization_statefulpartitionedcall_args_3<
8input_batch_normalization_statefulpartitionedcall_args_4)
%conv_1_statefulpartitionedcall_args_1)
%conv_1_statefulpartitionedcall_args_2+
'b_norm_1_statefulpartitionedcall_args_1+
'b_norm_1_statefulpartitionedcall_args_2+
'b_norm_1_statefulpartitionedcall_args_3+
'b_norm_1_statefulpartitionedcall_args_4)
%conv_2_statefulpartitionedcall_args_1)
%conv_2_statefulpartitionedcall_args_2+
'b_norm_2_statefulpartitionedcall_args_1+
'b_norm_2_statefulpartitionedcall_args_2+
'b_norm_2_statefulpartitionedcall_args_3+
'b_norm_2_statefulpartitionedcall_args_4)
%conv_3_statefulpartitionedcall_args_1)
%conv_3_statefulpartitionedcall_args_2+
'b_norm_3_statefulpartitionedcall_args_1+
'b_norm_3_statefulpartitionedcall_args_2+
'b_norm_3_statefulpartitionedcall_args_3+
'b_norm_3_statefulpartitionedcall_args_4)
%conv_4_statefulpartitionedcall_args_1)
%conv_4_statefulpartitionedcall_args_2+
'b_norm_4_statefulpartitionedcall_args_1+
'b_norm_4_statefulpartitionedcall_args_2+
'b_norm_4_statefulpartitionedcall_args_3+
'b_norm_4_statefulpartitionedcall_args_4<
8output_class_distribution_statefulpartitionedcall_args_1<
8output_class_distribution_statefulpartitionedcall_args_2
identity�� b_norm_1/StatefulPartitionedCall� b_norm_2/StatefulPartitionedCall� b_norm_3/StatefulPartitionedCall� b_norm_4/StatefulPartitionedCall�conv_1/StatefulPartitionedCall�conv_2/StatefulPartitionedCall�conv_3/StatefulPartitionedCall�conv_4/StatefulPartitionedCall�"dropout_42/StatefulPartitionedCall�"dropout_43/StatefulPartitionedCall�1input_batch_normalization/StatefulPartitionedCall�"input_conv/StatefulPartitionedCall�1output_class_distribution/StatefulPartitionedCall�
"input_conv/StatefulPartitionedCallStatefulPartitionedCallinput_conv_input)input_conv_statefulpartitionedcall_args_1)input_conv_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*1
_output_shapes
:�����������*-
config_proto

GPU

CPU2*0J 8*P
fKRI
G__inference_input_conv_layer_call_and_return_conditional_losses_43520442$
"input_conv/StatefulPartitionedCall�
1input_batch_normalization/StatefulPartitionedCallStatefulPartitionedCall+input_conv/StatefulPartitionedCall:output:08input_batch_normalization_statefulpartitionedcall_args_18input_batch_normalization_statefulpartitionedcall_args_28input_batch_normalization_statefulpartitionedcall_args_38input_batch_normalization_statefulpartitionedcall_args_4*
Tin	
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*1
_output_shapes
:�����������*-
config_proto

GPU

CPU2*0J 8*_
fZRX
V__inference_input_batch_normalization_layer_call_and_return_conditional_losses_435289223
1input_batch_normalization/StatefulPartitionedCall�
input_RELU/PartitionedCallPartitionedCall:input_batch_normalization/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*1
_output_shapes
:�����������*-
config_proto

GPU

CPU2*0J 8*P
fKRI
G__inference_input_RELU_layer_call_and_return_conditional_losses_43529432
input_RELU/PartitionedCall�
 max_pooling2d_85/PartitionedCallPartitionedCall#input_RELU/PartitionedCall:output:0*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*/
_output_shapes
:���������@@*-
config_proto

GPU

CPU2*0J 8*V
fQRO
M__inference_max_pooling2d_85_layer_call_and_return_conditional_losses_43521902"
 max_pooling2d_85/PartitionedCall�
"dropout_42/StatefulPartitionedCallStatefulPartitionedCall)max_pooling2d_85/PartitionedCall:output:0*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*/
_output_shapes
:���������@@*-
config_proto

GPU

CPU2*0J 8*P
fKRI
G__inference_dropout_42_layer_call_and_return_conditional_losses_43529722$
"dropout_42/StatefulPartitionedCall�
conv_1/StatefulPartitionedCallStatefulPartitionedCall+dropout_42/StatefulPartitionedCall:output:0%conv_1_statefulpartitionedcall_args_1%conv_1_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*/
_output_shapes
:���������@@*-
config_proto

GPU

CPU2*0J 8*L
fGRE
C__inference_conv_1_layer_call_and_return_conditional_losses_43522082 
conv_1/StatefulPartitionedCall�
 b_norm_1/StatefulPartitionedCallStatefulPartitionedCall'conv_1/StatefulPartitionedCall:output:0'b_norm_1_statefulpartitionedcall_args_1'b_norm_1_statefulpartitionedcall_args_2'b_norm_1_statefulpartitionedcall_args_3'b_norm_1_statefulpartitionedcall_args_4*
Tin	
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*/
_output_shapes
:���������@@*-
config_proto

GPU

CPU2*0J 8*N
fIRG
E__inference_b_norm_1_layer_call_and_return_conditional_losses_43530262"
 b_norm_1/StatefulPartitionedCall�
RELU_1/PartitionedCallPartitionedCall)b_norm_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*/
_output_shapes
:���������@@*-
config_proto

GPU

CPU2*0J 8*L
fGRE
C__inference_RELU_1_layer_call_and_return_conditional_losses_43530772
RELU_1/PartitionedCall�
 max_pooling2d_81/PartitionedCallPartitionedCallRELU_1/PartitionedCall:output:0*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*/
_output_shapes
:���������  *-
config_proto

GPU

CPU2*0J 8*V
fQRO
M__inference_max_pooling2d_81_layer_call_and_return_conditional_losses_43523542"
 max_pooling2d_81/PartitionedCall�
conv_2/StatefulPartitionedCallStatefulPartitionedCall)max_pooling2d_81/PartitionedCall:output:0%conv_2_statefulpartitionedcall_args_1%conv_2_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*/
_output_shapes
:���������   *-
config_proto

GPU

CPU2*0J 8*L
fGRE
C__inference_conv_2_layer_call_and_return_conditional_losses_43523722 
conv_2/StatefulPartitionedCall�
 b_norm_2/StatefulPartitionedCallStatefulPartitionedCall'conv_2/StatefulPartitionedCall:output:0'b_norm_2_statefulpartitionedcall_args_1'b_norm_2_statefulpartitionedcall_args_2'b_norm_2_statefulpartitionedcall_args_3'b_norm_2_statefulpartitionedcall_args_4*
Tin	
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*/
_output_shapes
:���������   *-
config_proto

GPU

CPU2*0J 8*N
fIRG
E__inference_b_norm_2_layer_call_and_return_conditional_losses_43531222"
 b_norm_2/StatefulPartitionedCall�
RELU_2/PartitionedCallPartitionedCall)b_norm_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*/
_output_shapes
:���������   *-
config_proto

GPU

CPU2*0J 8*L
fGRE
C__inference_RELU_2_layer_call_and_return_conditional_losses_43531732
RELU_2/PartitionedCall�
 max_pooling2d_82/PartitionedCallPartitionedCallRELU_2/PartitionedCall:output:0*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*/
_output_shapes
:��������� *-
config_proto

GPU

CPU2*0J 8*V
fQRO
M__inference_max_pooling2d_82_layer_call_and_return_conditional_losses_43525182"
 max_pooling2d_82/PartitionedCall�
conv_3/StatefulPartitionedCallStatefulPartitionedCall)max_pooling2d_82/PartitionedCall:output:0%conv_3_statefulpartitionedcall_args_1%conv_3_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*/
_output_shapes
:���������@*-
config_proto

GPU

CPU2*0J 8*L
fGRE
C__inference_conv_3_layer_call_and_return_conditional_losses_43525362 
conv_3/StatefulPartitionedCall�
 b_norm_3/StatefulPartitionedCallStatefulPartitionedCall'conv_3/StatefulPartitionedCall:output:0'b_norm_3_statefulpartitionedcall_args_1'b_norm_3_statefulpartitionedcall_args_2'b_norm_3_statefulpartitionedcall_args_3'b_norm_3_statefulpartitionedcall_args_4*
Tin	
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*/
_output_shapes
:���������@*-
config_proto

GPU

CPU2*0J 8*N
fIRG
E__inference_b_norm_3_layer_call_and_return_conditional_losses_43532182"
 b_norm_3/StatefulPartitionedCall�
RELU_3/PartitionedCallPartitionedCall)b_norm_3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*/
_output_shapes
:���������@*-
config_proto

GPU

CPU2*0J 8*L
fGRE
C__inference_RELU_3_layer_call_and_return_conditional_losses_43532692
RELU_3/PartitionedCall�
 max_pooling2d_83/PartitionedCallPartitionedCallRELU_3/PartitionedCall:output:0*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*/
_output_shapes
:���������@*-
config_proto

GPU

CPU2*0J 8*V
fQRO
M__inference_max_pooling2d_83_layer_call_and_return_conditional_losses_43526822"
 max_pooling2d_83/PartitionedCall�
conv_4/StatefulPartitionedCallStatefulPartitionedCall)max_pooling2d_83/PartitionedCall:output:0%conv_4_statefulpartitionedcall_args_1%conv_4_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*/
_output_shapes
:���������@*-
config_proto

GPU

CPU2*0J 8*L
fGRE
C__inference_conv_4_layer_call_and_return_conditional_losses_43527002 
conv_4/StatefulPartitionedCall�
 b_norm_4/StatefulPartitionedCallStatefulPartitionedCall'conv_4/StatefulPartitionedCall:output:0'b_norm_4_statefulpartitionedcall_args_1'b_norm_4_statefulpartitionedcall_args_2'b_norm_4_statefulpartitionedcall_args_3'b_norm_4_statefulpartitionedcall_args_4*
Tin	
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*/
_output_shapes
:���������@*-
config_proto

GPU

CPU2*0J 8*N
fIRG
E__inference_b_norm_4_layer_call_and_return_conditional_losses_43533142"
 b_norm_4/StatefulPartitionedCall�
RELU_4/PartitionedCallPartitionedCall)b_norm_4/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*/
_output_shapes
:���������@*-
config_proto

GPU

CPU2*0J 8*L
fGRE
C__inference_RELU_4_layer_call_and_return_conditional_losses_43533652
RELU_4/PartitionedCall�
 max_pooling2d_84/PartitionedCallPartitionedCallRELU_4/PartitionedCall:output:0*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*/
_output_shapes
:���������@*-
config_proto

GPU

CPU2*0J 8*V
fQRO
M__inference_max_pooling2d_84_layer_call_and_return_conditional_losses_43528462"
 max_pooling2d_84/PartitionedCall�
flatten_21/PartitionedCallPartitionedCall)max_pooling2d_84/PartitionedCall:output:0*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*(
_output_shapes
:����������*-
config_proto

GPU

CPU2*0J 8*P
fKRI
G__inference_flatten_21_layer_call_and_return_conditional_losses_43533802
flatten_21/PartitionedCall�
"dropout_43/StatefulPartitionedCallStatefulPartitionedCall#flatten_21/PartitionedCall:output:0#^dropout_42/StatefulPartitionedCall*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*(
_output_shapes
:����������*-
config_proto

GPU

CPU2*0J 8*P
fKRI
G__inference_dropout_43_layer_call_and_return_conditional_losses_43534082$
"dropout_43/StatefulPartitionedCall�
1output_class_distribution/StatefulPartitionedCallStatefulPartitionedCall+dropout_43/StatefulPartitionedCall:output:08output_class_distribution_statefulpartitionedcall_args_18output_class_distribution_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*(
_output_shapes
:����������*-
config_proto

GPU

CPU2*0J 8*_
fZRX
V__inference_output_class_distribution_layer_call_and_return_conditional_losses_435343623
1output_class_distribution/StatefulPartitionedCall�
IdentityIdentity:output_class_distribution/StatefulPartitionedCall:output:0!^b_norm_1/StatefulPartitionedCall!^b_norm_2/StatefulPartitionedCall!^b_norm_3/StatefulPartitionedCall!^b_norm_4/StatefulPartitionedCall^conv_1/StatefulPartitionedCall^conv_2/StatefulPartitionedCall^conv_3/StatefulPartitionedCall^conv_4/StatefulPartitionedCall#^dropout_42/StatefulPartitionedCall#^dropout_43/StatefulPartitionedCall2^input_batch_normalization/StatefulPartitionedCall#^input_conv/StatefulPartitionedCall2^output_class_distribution/StatefulPartitionedCall*
T0*(
_output_shapes
:����������2

Identity"
identityIdentity:output:0*�
_input_shapes�
�:�����������::::::::::::::::::::::::::::::::2D
 b_norm_1/StatefulPartitionedCall b_norm_1/StatefulPartitionedCall2D
 b_norm_2/StatefulPartitionedCall b_norm_2/StatefulPartitionedCall2D
 b_norm_3/StatefulPartitionedCall b_norm_3/StatefulPartitionedCall2D
 b_norm_4/StatefulPartitionedCall b_norm_4/StatefulPartitionedCall2@
conv_1/StatefulPartitionedCallconv_1/StatefulPartitionedCall2@
conv_2/StatefulPartitionedCallconv_2/StatefulPartitionedCall2@
conv_3/StatefulPartitionedCallconv_3/StatefulPartitionedCall2@
conv_4/StatefulPartitionedCallconv_4/StatefulPartitionedCall2H
"dropout_42/StatefulPartitionedCall"dropout_42/StatefulPartitionedCall2H
"dropout_43/StatefulPartitionedCall"dropout_43/StatefulPartitionedCall2f
1input_batch_normalization/StatefulPartitionedCall1input_batch_normalization/StatefulPartitionedCall2H
"input_conv/StatefulPartitionedCall"input_conv/StatefulPartitionedCall2f
1output_class_distribution/StatefulPartitionedCall1output_class_distribution/StatefulPartitionedCall:0 ,
*
_user_specified_nameinput_conv_input
�
�
;__inference_input_batch_normalization_layer_call_fn_4354413

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4*
Tin	
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*1
_output_shapes
:�����������*-
config_proto

GPU

CPU2*0J 8*_
fZRX
V__inference_input_batch_normalization_layer_call_and_return_conditional_losses_43528922
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*1
_output_shapes
:�����������2

Identity"
identityIdentity:output:0*@
_input_shapes/
-:�����������::::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs
�
e
G__inference_dropout_43_layer_call_and_return_conditional_losses_4355183

inputs

identity_1[
IdentityIdentityinputs*
T0*(
_output_shapes
:����������2

Identityj

Identity_1IdentityIdentity:output:0*
T0*(
_output_shapes
:����������2

Identity_1"!

identity_1Identity_1:output:0*'
_input_shapes
:����������:& "
 
_user_specified_nameinputs
�
_
C__inference_RELU_2_layer_call_and_return_conditional_losses_4353173

inputs
identityV
ReluReluinputs*
T0*/
_output_shapes
:���������   2
Relun
IdentityIdentityRelu:activations:0*
T0*/
_output_shapes
:���������   2

Identity"
identityIdentity:output:0*.
_input_shapes
:���������   :& "
 
_user_specified_nameinputs
�
N
2__inference_max_pooling2d_81_layer_call_fn_4352360

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*J
_output_shapes8
6:4������������������������������������*-
config_proto

GPU

CPU2*0J 8*V
fQRO
M__inference_max_pooling2d_81_layer_call_and_return_conditional_losses_43523542
PartitionedCall�
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4������������������������������������2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4������������������������������������:& "
 
_user_specified_nameinputs
�o
�
G__inference_awsome_net_layer_call_and_return_conditional_losses_4353669

inputs-
)input_conv_statefulpartitionedcall_args_1-
)input_conv_statefulpartitionedcall_args_2<
8input_batch_normalization_statefulpartitionedcall_args_1<
8input_batch_normalization_statefulpartitionedcall_args_2<
8input_batch_normalization_statefulpartitionedcall_args_3<
8input_batch_normalization_statefulpartitionedcall_args_4)
%conv_1_statefulpartitionedcall_args_1)
%conv_1_statefulpartitionedcall_args_2+
'b_norm_1_statefulpartitionedcall_args_1+
'b_norm_1_statefulpartitionedcall_args_2+
'b_norm_1_statefulpartitionedcall_args_3+
'b_norm_1_statefulpartitionedcall_args_4)
%conv_2_statefulpartitionedcall_args_1)
%conv_2_statefulpartitionedcall_args_2+
'b_norm_2_statefulpartitionedcall_args_1+
'b_norm_2_statefulpartitionedcall_args_2+
'b_norm_2_statefulpartitionedcall_args_3+
'b_norm_2_statefulpartitionedcall_args_4)
%conv_3_statefulpartitionedcall_args_1)
%conv_3_statefulpartitionedcall_args_2+
'b_norm_3_statefulpartitionedcall_args_1+
'b_norm_3_statefulpartitionedcall_args_2+
'b_norm_3_statefulpartitionedcall_args_3+
'b_norm_3_statefulpartitionedcall_args_4)
%conv_4_statefulpartitionedcall_args_1)
%conv_4_statefulpartitionedcall_args_2+
'b_norm_4_statefulpartitionedcall_args_1+
'b_norm_4_statefulpartitionedcall_args_2+
'b_norm_4_statefulpartitionedcall_args_3+
'b_norm_4_statefulpartitionedcall_args_4<
8output_class_distribution_statefulpartitionedcall_args_1<
8output_class_distribution_statefulpartitionedcall_args_2
identity�� b_norm_1/StatefulPartitionedCall� b_norm_2/StatefulPartitionedCall� b_norm_3/StatefulPartitionedCall� b_norm_4/StatefulPartitionedCall�conv_1/StatefulPartitionedCall�conv_2/StatefulPartitionedCall�conv_3/StatefulPartitionedCall�conv_4/StatefulPartitionedCall�1input_batch_normalization/StatefulPartitionedCall�"input_conv/StatefulPartitionedCall�1output_class_distribution/StatefulPartitionedCall�
"input_conv/StatefulPartitionedCallStatefulPartitionedCallinputs)input_conv_statefulpartitionedcall_args_1)input_conv_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*1
_output_shapes
:�����������*-
config_proto

GPU

CPU2*0J 8*P
fKRI
G__inference_input_conv_layer_call_and_return_conditional_losses_43520442$
"input_conv/StatefulPartitionedCall�
1input_batch_normalization/StatefulPartitionedCallStatefulPartitionedCall+input_conv/StatefulPartitionedCall:output:08input_batch_normalization_statefulpartitionedcall_args_18input_batch_normalization_statefulpartitionedcall_args_28input_batch_normalization_statefulpartitionedcall_args_38input_batch_normalization_statefulpartitionedcall_args_4*
Tin	
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*1
_output_shapes
:�����������*-
config_proto

GPU

CPU2*0J 8*_
fZRX
V__inference_input_batch_normalization_layer_call_and_return_conditional_losses_435291423
1input_batch_normalization/StatefulPartitionedCall�
input_RELU/PartitionedCallPartitionedCall:input_batch_normalization/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*1
_output_shapes
:�����������*-
config_proto

GPU

CPU2*0J 8*P
fKRI
G__inference_input_RELU_layer_call_and_return_conditional_losses_43529432
input_RELU/PartitionedCall�
 max_pooling2d_85/PartitionedCallPartitionedCall#input_RELU/PartitionedCall:output:0*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*/
_output_shapes
:���������@@*-
config_proto

GPU

CPU2*0J 8*V
fQRO
M__inference_max_pooling2d_85_layer_call_and_return_conditional_losses_43521902"
 max_pooling2d_85/PartitionedCall�
dropout_42/PartitionedCallPartitionedCall)max_pooling2d_85/PartitionedCall:output:0*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*/
_output_shapes
:���������@@*-
config_proto

GPU

CPU2*0J 8*P
fKRI
G__inference_dropout_42_layer_call_and_return_conditional_losses_43529772
dropout_42/PartitionedCall�
conv_1/StatefulPartitionedCallStatefulPartitionedCall#dropout_42/PartitionedCall:output:0%conv_1_statefulpartitionedcall_args_1%conv_1_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*/
_output_shapes
:���������@@*-
config_proto

GPU

CPU2*0J 8*L
fGRE
C__inference_conv_1_layer_call_and_return_conditional_losses_43522082 
conv_1/StatefulPartitionedCall�
 b_norm_1/StatefulPartitionedCallStatefulPartitionedCall'conv_1/StatefulPartitionedCall:output:0'b_norm_1_statefulpartitionedcall_args_1'b_norm_1_statefulpartitionedcall_args_2'b_norm_1_statefulpartitionedcall_args_3'b_norm_1_statefulpartitionedcall_args_4*
Tin	
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*/
_output_shapes
:���������@@*-
config_proto

GPU

CPU2*0J 8*N
fIRG
E__inference_b_norm_1_layer_call_and_return_conditional_losses_43530482"
 b_norm_1/StatefulPartitionedCall�
RELU_1/PartitionedCallPartitionedCall)b_norm_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*/
_output_shapes
:���������@@*-
config_proto

GPU

CPU2*0J 8*L
fGRE
C__inference_RELU_1_layer_call_and_return_conditional_losses_43530772
RELU_1/PartitionedCall�
 max_pooling2d_81/PartitionedCallPartitionedCallRELU_1/PartitionedCall:output:0*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*/
_output_shapes
:���������  *-
config_proto

GPU

CPU2*0J 8*V
fQRO
M__inference_max_pooling2d_81_layer_call_and_return_conditional_losses_43523542"
 max_pooling2d_81/PartitionedCall�
conv_2/StatefulPartitionedCallStatefulPartitionedCall)max_pooling2d_81/PartitionedCall:output:0%conv_2_statefulpartitionedcall_args_1%conv_2_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*/
_output_shapes
:���������   *-
config_proto

GPU

CPU2*0J 8*L
fGRE
C__inference_conv_2_layer_call_and_return_conditional_losses_43523722 
conv_2/StatefulPartitionedCall�
 b_norm_2/StatefulPartitionedCallStatefulPartitionedCall'conv_2/StatefulPartitionedCall:output:0'b_norm_2_statefulpartitionedcall_args_1'b_norm_2_statefulpartitionedcall_args_2'b_norm_2_statefulpartitionedcall_args_3'b_norm_2_statefulpartitionedcall_args_4*
Tin	
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*/
_output_shapes
:���������   *-
config_proto

GPU

CPU2*0J 8*N
fIRG
E__inference_b_norm_2_layer_call_and_return_conditional_losses_43531442"
 b_norm_2/StatefulPartitionedCall�
RELU_2/PartitionedCallPartitionedCall)b_norm_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*/
_output_shapes
:���������   *-
config_proto

GPU

CPU2*0J 8*L
fGRE
C__inference_RELU_2_layer_call_and_return_conditional_losses_43531732
RELU_2/PartitionedCall�
 max_pooling2d_82/PartitionedCallPartitionedCallRELU_2/PartitionedCall:output:0*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*/
_output_shapes
:��������� *-
config_proto

GPU

CPU2*0J 8*V
fQRO
M__inference_max_pooling2d_82_layer_call_and_return_conditional_losses_43525182"
 max_pooling2d_82/PartitionedCall�
conv_3/StatefulPartitionedCallStatefulPartitionedCall)max_pooling2d_82/PartitionedCall:output:0%conv_3_statefulpartitionedcall_args_1%conv_3_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*/
_output_shapes
:���������@*-
config_proto

GPU

CPU2*0J 8*L
fGRE
C__inference_conv_3_layer_call_and_return_conditional_losses_43525362 
conv_3/StatefulPartitionedCall�
 b_norm_3/StatefulPartitionedCallStatefulPartitionedCall'conv_3/StatefulPartitionedCall:output:0'b_norm_3_statefulpartitionedcall_args_1'b_norm_3_statefulpartitionedcall_args_2'b_norm_3_statefulpartitionedcall_args_3'b_norm_3_statefulpartitionedcall_args_4*
Tin	
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*/
_output_shapes
:���������@*-
config_proto

GPU

CPU2*0J 8*N
fIRG
E__inference_b_norm_3_layer_call_and_return_conditional_losses_43532402"
 b_norm_3/StatefulPartitionedCall�
RELU_3/PartitionedCallPartitionedCall)b_norm_3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*/
_output_shapes
:���������@*-
config_proto

GPU

CPU2*0J 8*L
fGRE
C__inference_RELU_3_layer_call_and_return_conditional_losses_43532692
RELU_3/PartitionedCall�
 max_pooling2d_83/PartitionedCallPartitionedCallRELU_3/PartitionedCall:output:0*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*/
_output_shapes
:���������@*-
config_proto

GPU

CPU2*0J 8*V
fQRO
M__inference_max_pooling2d_83_layer_call_and_return_conditional_losses_43526822"
 max_pooling2d_83/PartitionedCall�
conv_4/StatefulPartitionedCallStatefulPartitionedCall)max_pooling2d_83/PartitionedCall:output:0%conv_4_statefulpartitionedcall_args_1%conv_4_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*/
_output_shapes
:���������@*-
config_proto

GPU

CPU2*0J 8*L
fGRE
C__inference_conv_4_layer_call_and_return_conditional_losses_43527002 
conv_4/StatefulPartitionedCall�
 b_norm_4/StatefulPartitionedCallStatefulPartitionedCall'conv_4/StatefulPartitionedCall:output:0'b_norm_4_statefulpartitionedcall_args_1'b_norm_4_statefulpartitionedcall_args_2'b_norm_4_statefulpartitionedcall_args_3'b_norm_4_statefulpartitionedcall_args_4*
Tin	
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*/
_output_shapes
:���������@*-
config_proto

GPU

CPU2*0J 8*N
fIRG
E__inference_b_norm_4_layer_call_and_return_conditional_losses_43533362"
 b_norm_4/StatefulPartitionedCall�
RELU_4/PartitionedCallPartitionedCall)b_norm_4/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*/
_output_shapes
:���������@*-
config_proto

GPU

CPU2*0J 8*L
fGRE
C__inference_RELU_4_layer_call_and_return_conditional_losses_43533652
RELU_4/PartitionedCall�
 max_pooling2d_84/PartitionedCallPartitionedCallRELU_4/PartitionedCall:output:0*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*/
_output_shapes
:���������@*-
config_proto

GPU

CPU2*0J 8*V
fQRO
M__inference_max_pooling2d_84_layer_call_and_return_conditional_losses_43528462"
 max_pooling2d_84/PartitionedCall�
flatten_21/PartitionedCallPartitionedCall)max_pooling2d_84/PartitionedCall:output:0*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*(
_output_shapes
:����������*-
config_proto

GPU

CPU2*0J 8*P
fKRI
G__inference_flatten_21_layer_call_and_return_conditional_losses_43533802
flatten_21/PartitionedCall�
dropout_43/PartitionedCallPartitionedCall#flatten_21/PartitionedCall:output:0*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*(
_output_shapes
:����������*-
config_proto

GPU

CPU2*0J 8*P
fKRI
G__inference_dropout_43_layer_call_and_return_conditional_losses_43534132
dropout_43/PartitionedCall�
1output_class_distribution/StatefulPartitionedCallStatefulPartitionedCall#dropout_43/PartitionedCall:output:08output_class_distribution_statefulpartitionedcall_args_18output_class_distribution_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*(
_output_shapes
:����������*-
config_proto

GPU

CPU2*0J 8*_
fZRX
V__inference_output_class_distribution_layer_call_and_return_conditional_losses_435343623
1output_class_distribution/StatefulPartitionedCall�
IdentityIdentity:output_class_distribution/StatefulPartitionedCall:output:0!^b_norm_1/StatefulPartitionedCall!^b_norm_2/StatefulPartitionedCall!^b_norm_3/StatefulPartitionedCall!^b_norm_4/StatefulPartitionedCall^conv_1/StatefulPartitionedCall^conv_2/StatefulPartitionedCall^conv_3/StatefulPartitionedCall^conv_4/StatefulPartitionedCall2^input_batch_normalization/StatefulPartitionedCall#^input_conv/StatefulPartitionedCall2^output_class_distribution/StatefulPartitionedCall*
T0*(
_output_shapes
:����������2

Identity"
identityIdentity:output:0*�
_input_shapes�
�:�����������::::::::::::::::::::::::::::::::2D
 b_norm_1/StatefulPartitionedCall b_norm_1/StatefulPartitionedCall2D
 b_norm_2/StatefulPartitionedCall b_norm_2/StatefulPartitionedCall2D
 b_norm_3/StatefulPartitionedCall b_norm_3/StatefulPartitionedCall2D
 b_norm_4/StatefulPartitionedCall b_norm_4/StatefulPartitionedCall2@
conv_1/StatefulPartitionedCallconv_1/StatefulPartitionedCall2@
conv_2/StatefulPartitionedCallconv_2/StatefulPartitionedCall2@
conv_3/StatefulPartitionedCallconv_3/StatefulPartitionedCall2@
conv_4/StatefulPartitionedCallconv_4/StatefulPartitionedCall2f
1input_batch_normalization/StatefulPartitionedCall1input_batch_normalization/StatefulPartitionedCall2H
"input_conv/StatefulPartitionedCall"input_conv/StatefulPartitionedCall2f
1output_class_distribution/StatefulPartitionedCall1output_class_distribution/StatefulPartitionedCall:& "
 
_user_specified_nameinputs
�
�
(__inference_conv_2_layer_call_fn_4352380

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*A
_output_shapes/
-:+��������������������������� *-
config_proto

GPU

CPU2*0J 8*L
fGRE
C__inference_conv_2_layer_call_and_return_conditional_losses_43523722
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+��������������������������� 2

Identity"
identityIdentity:output:0*H
_input_shapes7
5:+���������������������������::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs
�
�
V__inference_input_batch_normalization_layer_call_and_return_conditional_losses_4354404

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity��FusedBatchNormV3/ReadVariableOp�!FusedBatchNormV3/ReadVariableOp_1�ReadVariableOp�ReadVariableOp_1^
LogicalAnd/xConst*
_output_shapes
: *
dtype0
*
value	B
 Z 2
LogicalAnd/x^
LogicalAnd/yConst*
_output_shapes
: *
dtype0
*
value	B
 Z2
LogicalAnd/yl

LogicalAnd
LogicalAndLogicalAnd/x:output:0LogicalAnd/y:output:0*
_output_shapes
: 2

LogicalAndt
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOp�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*M
_output_shapes;
9:�����������:::::*
epsilon%o�:*
is_training( 2
FusedBatchNormV3S
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *�p}?2
Const�
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*1
_output_shapes
:�����������2

Identity"
identityIdentity:output:0*@
_input_shapes/
-:�����������::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:& "
 
_user_specified_nameinputs
�$
�
E__inference_b_norm_3_layer_call_and_return_conditional_losses_4354927

inputs
readvariableop_resource
readvariableop_1_resource
assignmovingavg_4354912
assignmovingavg_1_4354919
identity��#AssignMovingAvg/AssignSubVariableOp�AssignMovingAvg/ReadVariableOp�%AssignMovingAvg_1/AssignSubVariableOp� AssignMovingAvg_1/ReadVariableOp�ReadVariableOp�ReadVariableOp_1^
LogicalAnd/xConst*
_output_shapes
: *
dtype0
*
value	B
 Z2
LogicalAnd/x^
LogicalAnd/yConst*
_output_shapes
: *
dtype0
*
value	B
 Z2
LogicalAnd/yl

LogicalAnd
LogicalAndLogicalAnd/x:output:0LogicalAnd/y:output:0*
_output_shapes
: 2

LogicalAndt
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1Q
ConstConst*
_output_shapes
: *
dtype0*
valueB 2
ConstU
Const_1Const*
_output_shapes
: *
dtype0*
valueB 2	
Const_1�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0Const:output:0Const_1:output:0*
T0*
U0*K
_output_shapes9
7:���������@:@:@:@:@:*
epsilon%o�:2
FusedBatchNormV3W
Const_2Const*
_output_shapes
: *
dtype0*
valueB
 *�p}?2	
Const_2�
AssignMovingAvg/sub/xConst**
_class 
loc:@AssignMovingAvg/4354912*
_output_shapes
: *
dtype0*
valueB
 *  �?2
AssignMovingAvg/sub/x�
AssignMovingAvg/subSubAssignMovingAvg/sub/x:output:0Const_2:output:0*
T0**
_class 
loc:@AssignMovingAvg/4354912*
_output_shapes
: 2
AssignMovingAvg/sub�
AssignMovingAvg/ReadVariableOpReadVariableOpassignmovingavg_4354912*
_output_shapes
:@*
dtype02 
AssignMovingAvg/ReadVariableOp�
AssignMovingAvg/sub_1Sub&AssignMovingAvg/ReadVariableOp:value:0FusedBatchNormV3:batch_mean:0*
T0**
_class 
loc:@AssignMovingAvg/4354912*
_output_shapes
:@2
AssignMovingAvg/sub_1�
AssignMovingAvg/mulMulAssignMovingAvg/sub_1:z:0AssignMovingAvg/sub:z:0*
T0**
_class 
loc:@AssignMovingAvg/4354912*
_output_shapes
:@2
AssignMovingAvg/mul�
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOpassignmovingavg_4354912AssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp**
_class 
loc:@AssignMovingAvg/4354912*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOp�
AssignMovingAvg_1/sub/xConst*,
_class"
 loc:@AssignMovingAvg_1/4354919*
_output_shapes
: *
dtype0*
valueB
 *  �?2
AssignMovingAvg_1/sub/x�
AssignMovingAvg_1/subSub AssignMovingAvg_1/sub/x:output:0Const_2:output:0*
T0*,
_class"
 loc:@AssignMovingAvg_1/4354919*
_output_shapes
: 2
AssignMovingAvg_1/sub�
 AssignMovingAvg_1/ReadVariableOpReadVariableOpassignmovingavg_1_4354919*
_output_shapes
:@*
dtype02"
 AssignMovingAvg_1/ReadVariableOp�
AssignMovingAvg_1/sub_1Sub(AssignMovingAvg_1/ReadVariableOp:value:0!FusedBatchNormV3:batch_variance:0*
T0*,
_class"
 loc:@AssignMovingAvg_1/4354919*
_output_shapes
:@2
AssignMovingAvg_1/sub_1�
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub_1:z:0AssignMovingAvg_1/sub:z:0*
T0*,
_class"
 loc:@AssignMovingAvg_1/4354919*
_output_shapes
:@2
AssignMovingAvg_1/mul�
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOpassignmovingavg_1_4354919AssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*,
_class"
 loc:@AssignMovingAvg_1/4354919*
_output_shapes
 *
dtype02'
%AssignMovingAvg_1/AssignSubVariableOp�
IdentityIdentityFusedBatchNormV3:y:0$^AssignMovingAvg/AssignSubVariableOp^AssignMovingAvg/ReadVariableOp&^AssignMovingAvg_1/AssignSubVariableOp!^AssignMovingAvg_1/ReadVariableOp^ReadVariableOp^ReadVariableOp_1*
T0*/
_output_shapes
:���������@2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:���������@::::2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp2D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp2 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:& "
 
_user_specified_nameinputs
��
�.
#__inference__traced_restore_4355744
file_prefix)
%assignvariableop_input_conv_21_kernel)
%assignvariableop_1_input_conv_21_bias9
5assignvariableop_2_input_batch_normalization_21_gamma8
4assignvariableop_3_input_batch_normalization_21_beta?
;assignvariableop_4_input_batch_normalization_21_moving_meanC
?assignvariableop_5_input_batch_normalization_21_moving_variance'
#assignvariableop_6_conv_1_21_kernel%
!assignvariableop_7_conv_1_21_bias(
$assignvariableop_8_b_norm_1_21_gamma'
#assignvariableop_9_b_norm_1_21_beta/
+assignvariableop_10_b_norm_1_21_moving_mean3
/assignvariableop_11_b_norm_1_21_moving_variance(
$assignvariableop_12_conv_2_21_kernel&
"assignvariableop_13_conv_2_21_bias)
%assignvariableop_14_b_norm_2_21_gamma(
$assignvariableop_15_b_norm_2_21_beta/
+assignvariableop_16_b_norm_2_21_moving_mean3
/assignvariableop_17_b_norm_2_21_moving_variance(
$assignvariableop_18_conv_3_13_kernel&
"assignvariableop_19_conv_3_13_bias)
%assignvariableop_20_b_norm_3_13_gamma(
$assignvariableop_21_b_norm_3_13_beta/
+assignvariableop_22_b_norm_3_13_moving_mean3
/assignvariableop_23_b_norm_3_13_moving_variance'
#assignvariableop_24_conv_4_5_kernel%
!assignvariableop_25_conv_4_5_bias(
$assignvariableop_26_b_norm_4_5_gamma'
#assignvariableop_27_b_norm_4_5_beta.
*assignvariableop_28_b_norm_4_5_moving_mean2
.assignvariableop_29_b_norm_4_5_moving_variance;
7assignvariableop_30_output_class_distribution_21_kernel9
5assignvariableop_31_output_class_distribution_21_bias!
assignvariableop_32_adam_iter#
assignvariableop_33_adam_beta_1#
assignvariableop_34_adam_beta_2"
assignvariableop_35_adam_decay*
&assignvariableop_36_adam_learning_rate
assignvariableop_37_total
assignvariableop_38_count3
/assignvariableop_39_adam_input_conv_21_kernel_m1
-assignvariableop_40_adam_input_conv_21_bias_mA
=assignvariableop_41_adam_input_batch_normalization_21_gamma_m@
<assignvariableop_42_adam_input_batch_normalization_21_beta_m/
+assignvariableop_43_adam_conv_1_21_kernel_m-
)assignvariableop_44_adam_conv_1_21_bias_m0
,assignvariableop_45_adam_b_norm_1_21_gamma_m/
+assignvariableop_46_adam_b_norm_1_21_beta_m/
+assignvariableop_47_adam_conv_2_21_kernel_m-
)assignvariableop_48_adam_conv_2_21_bias_m0
,assignvariableop_49_adam_b_norm_2_21_gamma_m/
+assignvariableop_50_adam_b_norm_2_21_beta_m/
+assignvariableop_51_adam_conv_3_13_kernel_m-
)assignvariableop_52_adam_conv_3_13_bias_m0
,assignvariableop_53_adam_b_norm_3_13_gamma_m/
+assignvariableop_54_adam_b_norm_3_13_beta_m.
*assignvariableop_55_adam_conv_4_5_kernel_m,
(assignvariableop_56_adam_conv_4_5_bias_m/
+assignvariableop_57_adam_b_norm_4_5_gamma_m.
*assignvariableop_58_adam_b_norm_4_5_beta_mB
>assignvariableop_59_adam_output_class_distribution_21_kernel_m@
<assignvariableop_60_adam_output_class_distribution_21_bias_m3
/assignvariableop_61_adam_input_conv_21_kernel_v1
-assignvariableop_62_adam_input_conv_21_bias_vA
=assignvariableop_63_adam_input_batch_normalization_21_gamma_v@
<assignvariableop_64_adam_input_batch_normalization_21_beta_v/
+assignvariableop_65_adam_conv_1_21_kernel_v-
)assignvariableop_66_adam_conv_1_21_bias_v0
,assignvariableop_67_adam_b_norm_1_21_gamma_v/
+assignvariableop_68_adam_b_norm_1_21_beta_v/
+assignvariableop_69_adam_conv_2_21_kernel_v-
)assignvariableop_70_adam_conv_2_21_bias_v0
,assignvariableop_71_adam_b_norm_2_21_gamma_v/
+assignvariableop_72_adam_b_norm_2_21_beta_v/
+assignvariableop_73_adam_conv_3_13_kernel_v-
)assignvariableop_74_adam_conv_3_13_bias_v0
,assignvariableop_75_adam_b_norm_3_13_gamma_v/
+assignvariableop_76_adam_b_norm_3_13_beta_v.
*assignvariableop_77_adam_conv_4_5_kernel_v,
(assignvariableop_78_adam_conv_4_5_bias_v/
+assignvariableop_79_adam_b_norm_4_5_gamma_v.
*assignvariableop_80_adam_b_norm_4_5_beta_vB
>assignvariableop_81_adam_output_class_distribution_21_kernel_v@
<assignvariableop_82_adam_output_class_distribution_21_bias_v
identity_84��AssignVariableOp�AssignVariableOp_1�AssignVariableOp_10�AssignVariableOp_11�AssignVariableOp_12�AssignVariableOp_13�AssignVariableOp_14�AssignVariableOp_15�AssignVariableOp_16�AssignVariableOp_17�AssignVariableOp_18�AssignVariableOp_19�AssignVariableOp_2�AssignVariableOp_20�AssignVariableOp_21�AssignVariableOp_22�AssignVariableOp_23�AssignVariableOp_24�AssignVariableOp_25�AssignVariableOp_26�AssignVariableOp_27�AssignVariableOp_28�AssignVariableOp_29�AssignVariableOp_3�AssignVariableOp_30�AssignVariableOp_31�AssignVariableOp_32�AssignVariableOp_33�AssignVariableOp_34�AssignVariableOp_35�AssignVariableOp_36�AssignVariableOp_37�AssignVariableOp_38�AssignVariableOp_39�AssignVariableOp_4�AssignVariableOp_40�AssignVariableOp_41�AssignVariableOp_42�AssignVariableOp_43�AssignVariableOp_44�AssignVariableOp_45�AssignVariableOp_46�AssignVariableOp_47�AssignVariableOp_48�AssignVariableOp_49�AssignVariableOp_5�AssignVariableOp_50�AssignVariableOp_51�AssignVariableOp_52�AssignVariableOp_53�AssignVariableOp_54�AssignVariableOp_55�AssignVariableOp_56�AssignVariableOp_57�AssignVariableOp_58�AssignVariableOp_59�AssignVariableOp_6�AssignVariableOp_60�AssignVariableOp_61�AssignVariableOp_62�AssignVariableOp_63�AssignVariableOp_64�AssignVariableOp_65�AssignVariableOp_66�AssignVariableOp_67�AssignVariableOp_68�AssignVariableOp_69�AssignVariableOp_7�AssignVariableOp_70�AssignVariableOp_71�AssignVariableOp_72�AssignVariableOp_73�AssignVariableOp_74�AssignVariableOp_75�AssignVariableOp_76�AssignVariableOp_77�AssignVariableOp_78�AssignVariableOp_79�AssignVariableOp_8�AssignVariableOp_80�AssignVariableOp_81�AssignVariableOp_82�AssignVariableOp_9�	RestoreV2�RestoreV2_1�.
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:S*
dtype0*�-
value�-B�-SB6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-1/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-1/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-1/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-3/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-3/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-3/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-5/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-5/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-5/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-7/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-7/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-7/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-7/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-8/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-8/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-9/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-9/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-9/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-9/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-10/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-10/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-1/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-3/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-5/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-7/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-7/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-8/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-8/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-9/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-9/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-10/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-10/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-1/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-3/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-5/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-7/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-7/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-8/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-8/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-9/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-9/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-10/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-10/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE2
RestoreV2/tensor_names�
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:S*
dtype0*�
value�B�SB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
RestoreV2/shape_and_slices�
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*�
_output_shapes�
�:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::*a
dtypesW
U2S	2
	RestoreV2X
IdentityIdentityRestoreV2:tensors:0*
T0*
_output_shapes
:2

Identity�
AssignVariableOpAssignVariableOp%assignvariableop_input_conv_21_kernelIdentity:output:0*
_output_shapes
 *
dtype02
AssignVariableOp\

Identity_1IdentityRestoreV2:tensors:1*
T0*
_output_shapes
:2

Identity_1�
AssignVariableOp_1AssignVariableOp%assignvariableop_1_input_conv_21_biasIdentity_1:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_1\

Identity_2IdentityRestoreV2:tensors:2*
T0*
_output_shapes
:2

Identity_2�
AssignVariableOp_2AssignVariableOp5assignvariableop_2_input_batch_normalization_21_gammaIdentity_2:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_2\

Identity_3IdentityRestoreV2:tensors:3*
T0*
_output_shapes
:2

Identity_3�
AssignVariableOp_3AssignVariableOp4assignvariableop_3_input_batch_normalization_21_betaIdentity_3:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_3\

Identity_4IdentityRestoreV2:tensors:4*
T0*
_output_shapes
:2

Identity_4�
AssignVariableOp_4AssignVariableOp;assignvariableop_4_input_batch_normalization_21_moving_meanIdentity_4:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_4\

Identity_5IdentityRestoreV2:tensors:5*
T0*
_output_shapes
:2

Identity_5�
AssignVariableOp_5AssignVariableOp?assignvariableop_5_input_batch_normalization_21_moving_varianceIdentity_5:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_5\

Identity_6IdentityRestoreV2:tensors:6*
T0*
_output_shapes
:2

Identity_6�
AssignVariableOp_6AssignVariableOp#assignvariableop_6_conv_1_21_kernelIdentity_6:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_6\

Identity_7IdentityRestoreV2:tensors:7*
T0*
_output_shapes
:2

Identity_7�
AssignVariableOp_7AssignVariableOp!assignvariableop_7_conv_1_21_biasIdentity_7:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_7\

Identity_8IdentityRestoreV2:tensors:8*
T0*
_output_shapes
:2

Identity_8�
AssignVariableOp_8AssignVariableOp$assignvariableop_8_b_norm_1_21_gammaIdentity_8:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_8\

Identity_9IdentityRestoreV2:tensors:9*
T0*
_output_shapes
:2

Identity_9�
AssignVariableOp_9AssignVariableOp#assignvariableop_9_b_norm_1_21_betaIdentity_9:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_9_
Identity_10IdentityRestoreV2:tensors:10*
T0*
_output_shapes
:2
Identity_10�
AssignVariableOp_10AssignVariableOp+assignvariableop_10_b_norm_1_21_moving_meanIdentity_10:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_10_
Identity_11IdentityRestoreV2:tensors:11*
T0*
_output_shapes
:2
Identity_11�
AssignVariableOp_11AssignVariableOp/assignvariableop_11_b_norm_1_21_moving_varianceIdentity_11:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_11_
Identity_12IdentityRestoreV2:tensors:12*
T0*
_output_shapes
:2
Identity_12�
AssignVariableOp_12AssignVariableOp$assignvariableop_12_conv_2_21_kernelIdentity_12:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_12_
Identity_13IdentityRestoreV2:tensors:13*
T0*
_output_shapes
:2
Identity_13�
AssignVariableOp_13AssignVariableOp"assignvariableop_13_conv_2_21_biasIdentity_13:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_13_
Identity_14IdentityRestoreV2:tensors:14*
T0*
_output_shapes
:2
Identity_14�
AssignVariableOp_14AssignVariableOp%assignvariableop_14_b_norm_2_21_gammaIdentity_14:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_14_
Identity_15IdentityRestoreV2:tensors:15*
T0*
_output_shapes
:2
Identity_15�
AssignVariableOp_15AssignVariableOp$assignvariableop_15_b_norm_2_21_betaIdentity_15:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_15_
Identity_16IdentityRestoreV2:tensors:16*
T0*
_output_shapes
:2
Identity_16�
AssignVariableOp_16AssignVariableOp+assignvariableop_16_b_norm_2_21_moving_meanIdentity_16:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_16_
Identity_17IdentityRestoreV2:tensors:17*
T0*
_output_shapes
:2
Identity_17�
AssignVariableOp_17AssignVariableOp/assignvariableop_17_b_norm_2_21_moving_varianceIdentity_17:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_17_
Identity_18IdentityRestoreV2:tensors:18*
T0*
_output_shapes
:2
Identity_18�
AssignVariableOp_18AssignVariableOp$assignvariableop_18_conv_3_13_kernelIdentity_18:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_18_
Identity_19IdentityRestoreV2:tensors:19*
T0*
_output_shapes
:2
Identity_19�
AssignVariableOp_19AssignVariableOp"assignvariableop_19_conv_3_13_biasIdentity_19:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_19_
Identity_20IdentityRestoreV2:tensors:20*
T0*
_output_shapes
:2
Identity_20�
AssignVariableOp_20AssignVariableOp%assignvariableop_20_b_norm_3_13_gammaIdentity_20:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_20_
Identity_21IdentityRestoreV2:tensors:21*
T0*
_output_shapes
:2
Identity_21�
AssignVariableOp_21AssignVariableOp$assignvariableop_21_b_norm_3_13_betaIdentity_21:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_21_
Identity_22IdentityRestoreV2:tensors:22*
T0*
_output_shapes
:2
Identity_22�
AssignVariableOp_22AssignVariableOp+assignvariableop_22_b_norm_3_13_moving_meanIdentity_22:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_22_
Identity_23IdentityRestoreV2:tensors:23*
T0*
_output_shapes
:2
Identity_23�
AssignVariableOp_23AssignVariableOp/assignvariableop_23_b_norm_3_13_moving_varianceIdentity_23:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_23_
Identity_24IdentityRestoreV2:tensors:24*
T0*
_output_shapes
:2
Identity_24�
AssignVariableOp_24AssignVariableOp#assignvariableop_24_conv_4_5_kernelIdentity_24:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_24_
Identity_25IdentityRestoreV2:tensors:25*
T0*
_output_shapes
:2
Identity_25�
AssignVariableOp_25AssignVariableOp!assignvariableop_25_conv_4_5_biasIdentity_25:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_25_
Identity_26IdentityRestoreV2:tensors:26*
T0*
_output_shapes
:2
Identity_26�
AssignVariableOp_26AssignVariableOp$assignvariableop_26_b_norm_4_5_gammaIdentity_26:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_26_
Identity_27IdentityRestoreV2:tensors:27*
T0*
_output_shapes
:2
Identity_27�
AssignVariableOp_27AssignVariableOp#assignvariableop_27_b_norm_4_5_betaIdentity_27:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_27_
Identity_28IdentityRestoreV2:tensors:28*
T0*
_output_shapes
:2
Identity_28�
AssignVariableOp_28AssignVariableOp*assignvariableop_28_b_norm_4_5_moving_meanIdentity_28:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_28_
Identity_29IdentityRestoreV2:tensors:29*
T0*
_output_shapes
:2
Identity_29�
AssignVariableOp_29AssignVariableOp.assignvariableop_29_b_norm_4_5_moving_varianceIdentity_29:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_29_
Identity_30IdentityRestoreV2:tensors:30*
T0*
_output_shapes
:2
Identity_30�
AssignVariableOp_30AssignVariableOp7assignvariableop_30_output_class_distribution_21_kernelIdentity_30:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_30_
Identity_31IdentityRestoreV2:tensors:31*
T0*
_output_shapes
:2
Identity_31�
AssignVariableOp_31AssignVariableOp5assignvariableop_31_output_class_distribution_21_biasIdentity_31:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_31_
Identity_32IdentityRestoreV2:tensors:32*
T0	*
_output_shapes
:2
Identity_32�
AssignVariableOp_32AssignVariableOpassignvariableop_32_adam_iterIdentity_32:output:0*
_output_shapes
 *
dtype0	2
AssignVariableOp_32_
Identity_33IdentityRestoreV2:tensors:33*
T0*
_output_shapes
:2
Identity_33�
AssignVariableOp_33AssignVariableOpassignvariableop_33_adam_beta_1Identity_33:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_33_
Identity_34IdentityRestoreV2:tensors:34*
T0*
_output_shapes
:2
Identity_34�
AssignVariableOp_34AssignVariableOpassignvariableop_34_adam_beta_2Identity_34:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_34_
Identity_35IdentityRestoreV2:tensors:35*
T0*
_output_shapes
:2
Identity_35�
AssignVariableOp_35AssignVariableOpassignvariableop_35_adam_decayIdentity_35:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_35_
Identity_36IdentityRestoreV2:tensors:36*
T0*
_output_shapes
:2
Identity_36�
AssignVariableOp_36AssignVariableOp&assignvariableop_36_adam_learning_rateIdentity_36:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_36_
Identity_37IdentityRestoreV2:tensors:37*
T0*
_output_shapes
:2
Identity_37�
AssignVariableOp_37AssignVariableOpassignvariableop_37_totalIdentity_37:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_37_
Identity_38IdentityRestoreV2:tensors:38*
T0*
_output_shapes
:2
Identity_38�
AssignVariableOp_38AssignVariableOpassignvariableop_38_countIdentity_38:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_38_
Identity_39IdentityRestoreV2:tensors:39*
T0*
_output_shapes
:2
Identity_39�
AssignVariableOp_39AssignVariableOp/assignvariableop_39_adam_input_conv_21_kernel_mIdentity_39:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_39_
Identity_40IdentityRestoreV2:tensors:40*
T0*
_output_shapes
:2
Identity_40�
AssignVariableOp_40AssignVariableOp-assignvariableop_40_adam_input_conv_21_bias_mIdentity_40:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_40_
Identity_41IdentityRestoreV2:tensors:41*
T0*
_output_shapes
:2
Identity_41�
AssignVariableOp_41AssignVariableOp=assignvariableop_41_adam_input_batch_normalization_21_gamma_mIdentity_41:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_41_
Identity_42IdentityRestoreV2:tensors:42*
T0*
_output_shapes
:2
Identity_42�
AssignVariableOp_42AssignVariableOp<assignvariableop_42_adam_input_batch_normalization_21_beta_mIdentity_42:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_42_
Identity_43IdentityRestoreV2:tensors:43*
T0*
_output_shapes
:2
Identity_43�
AssignVariableOp_43AssignVariableOp+assignvariableop_43_adam_conv_1_21_kernel_mIdentity_43:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_43_
Identity_44IdentityRestoreV2:tensors:44*
T0*
_output_shapes
:2
Identity_44�
AssignVariableOp_44AssignVariableOp)assignvariableop_44_adam_conv_1_21_bias_mIdentity_44:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_44_
Identity_45IdentityRestoreV2:tensors:45*
T0*
_output_shapes
:2
Identity_45�
AssignVariableOp_45AssignVariableOp,assignvariableop_45_adam_b_norm_1_21_gamma_mIdentity_45:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_45_
Identity_46IdentityRestoreV2:tensors:46*
T0*
_output_shapes
:2
Identity_46�
AssignVariableOp_46AssignVariableOp+assignvariableop_46_adam_b_norm_1_21_beta_mIdentity_46:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_46_
Identity_47IdentityRestoreV2:tensors:47*
T0*
_output_shapes
:2
Identity_47�
AssignVariableOp_47AssignVariableOp+assignvariableop_47_adam_conv_2_21_kernel_mIdentity_47:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_47_
Identity_48IdentityRestoreV2:tensors:48*
T0*
_output_shapes
:2
Identity_48�
AssignVariableOp_48AssignVariableOp)assignvariableop_48_adam_conv_2_21_bias_mIdentity_48:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_48_
Identity_49IdentityRestoreV2:tensors:49*
T0*
_output_shapes
:2
Identity_49�
AssignVariableOp_49AssignVariableOp,assignvariableop_49_adam_b_norm_2_21_gamma_mIdentity_49:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_49_
Identity_50IdentityRestoreV2:tensors:50*
T0*
_output_shapes
:2
Identity_50�
AssignVariableOp_50AssignVariableOp+assignvariableop_50_adam_b_norm_2_21_beta_mIdentity_50:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_50_
Identity_51IdentityRestoreV2:tensors:51*
T0*
_output_shapes
:2
Identity_51�
AssignVariableOp_51AssignVariableOp+assignvariableop_51_adam_conv_3_13_kernel_mIdentity_51:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_51_
Identity_52IdentityRestoreV2:tensors:52*
T0*
_output_shapes
:2
Identity_52�
AssignVariableOp_52AssignVariableOp)assignvariableop_52_adam_conv_3_13_bias_mIdentity_52:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_52_
Identity_53IdentityRestoreV2:tensors:53*
T0*
_output_shapes
:2
Identity_53�
AssignVariableOp_53AssignVariableOp,assignvariableop_53_adam_b_norm_3_13_gamma_mIdentity_53:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_53_
Identity_54IdentityRestoreV2:tensors:54*
T0*
_output_shapes
:2
Identity_54�
AssignVariableOp_54AssignVariableOp+assignvariableop_54_adam_b_norm_3_13_beta_mIdentity_54:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_54_
Identity_55IdentityRestoreV2:tensors:55*
T0*
_output_shapes
:2
Identity_55�
AssignVariableOp_55AssignVariableOp*assignvariableop_55_adam_conv_4_5_kernel_mIdentity_55:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_55_
Identity_56IdentityRestoreV2:tensors:56*
T0*
_output_shapes
:2
Identity_56�
AssignVariableOp_56AssignVariableOp(assignvariableop_56_adam_conv_4_5_bias_mIdentity_56:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_56_
Identity_57IdentityRestoreV2:tensors:57*
T0*
_output_shapes
:2
Identity_57�
AssignVariableOp_57AssignVariableOp+assignvariableop_57_adam_b_norm_4_5_gamma_mIdentity_57:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_57_
Identity_58IdentityRestoreV2:tensors:58*
T0*
_output_shapes
:2
Identity_58�
AssignVariableOp_58AssignVariableOp*assignvariableop_58_adam_b_norm_4_5_beta_mIdentity_58:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_58_
Identity_59IdentityRestoreV2:tensors:59*
T0*
_output_shapes
:2
Identity_59�
AssignVariableOp_59AssignVariableOp>assignvariableop_59_adam_output_class_distribution_21_kernel_mIdentity_59:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_59_
Identity_60IdentityRestoreV2:tensors:60*
T0*
_output_shapes
:2
Identity_60�
AssignVariableOp_60AssignVariableOp<assignvariableop_60_adam_output_class_distribution_21_bias_mIdentity_60:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_60_
Identity_61IdentityRestoreV2:tensors:61*
T0*
_output_shapes
:2
Identity_61�
AssignVariableOp_61AssignVariableOp/assignvariableop_61_adam_input_conv_21_kernel_vIdentity_61:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_61_
Identity_62IdentityRestoreV2:tensors:62*
T0*
_output_shapes
:2
Identity_62�
AssignVariableOp_62AssignVariableOp-assignvariableop_62_adam_input_conv_21_bias_vIdentity_62:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_62_
Identity_63IdentityRestoreV2:tensors:63*
T0*
_output_shapes
:2
Identity_63�
AssignVariableOp_63AssignVariableOp=assignvariableop_63_adam_input_batch_normalization_21_gamma_vIdentity_63:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_63_
Identity_64IdentityRestoreV2:tensors:64*
T0*
_output_shapes
:2
Identity_64�
AssignVariableOp_64AssignVariableOp<assignvariableop_64_adam_input_batch_normalization_21_beta_vIdentity_64:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_64_
Identity_65IdentityRestoreV2:tensors:65*
T0*
_output_shapes
:2
Identity_65�
AssignVariableOp_65AssignVariableOp+assignvariableop_65_adam_conv_1_21_kernel_vIdentity_65:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_65_
Identity_66IdentityRestoreV2:tensors:66*
T0*
_output_shapes
:2
Identity_66�
AssignVariableOp_66AssignVariableOp)assignvariableop_66_adam_conv_1_21_bias_vIdentity_66:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_66_
Identity_67IdentityRestoreV2:tensors:67*
T0*
_output_shapes
:2
Identity_67�
AssignVariableOp_67AssignVariableOp,assignvariableop_67_adam_b_norm_1_21_gamma_vIdentity_67:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_67_
Identity_68IdentityRestoreV2:tensors:68*
T0*
_output_shapes
:2
Identity_68�
AssignVariableOp_68AssignVariableOp+assignvariableop_68_adam_b_norm_1_21_beta_vIdentity_68:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_68_
Identity_69IdentityRestoreV2:tensors:69*
T0*
_output_shapes
:2
Identity_69�
AssignVariableOp_69AssignVariableOp+assignvariableop_69_adam_conv_2_21_kernel_vIdentity_69:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_69_
Identity_70IdentityRestoreV2:tensors:70*
T0*
_output_shapes
:2
Identity_70�
AssignVariableOp_70AssignVariableOp)assignvariableop_70_adam_conv_2_21_bias_vIdentity_70:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_70_
Identity_71IdentityRestoreV2:tensors:71*
T0*
_output_shapes
:2
Identity_71�
AssignVariableOp_71AssignVariableOp,assignvariableop_71_adam_b_norm_2_21_gamma_vIdentity_71:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_71_
Identity_72IdentityRestoreV2:tensors:72*
T0*
_output_shapes
:2
Identity_72�
AssignVariableOp_72AssignVariableOp+assignvariableop_72_adam_b_norm_2_21_beta_vIdentity_72:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_72_
Identity_73IdentityRestoreV2:tensors:73*
T0*
_output_shapes
:2
Identity_73�
AssignVariableOp_73AssignVariableOp+assignvariableop_73_adam_conv_3_13_kernel_vIdentity_73:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_73_
Identity_74IdentityRestoreV2:tensors:74*
T0*
_output_shapes
:2
Identity_74�
AssignVariableOp_74AssignVariableOp)assignvariableop_74_adam_conv_3_13_bias_vIdentity_74:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_74_
Identity_75IdentityRestoreV2:tensors:75*
T0*
_output_shapes
:2
Identity_75�
AssignVariableOp_75AssignVariableOp,assignvariableop_75_adam_b_norm_3_13_gamma_vIdentity_75:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_75_
Identity_76IdentityRestoreV2:tensors:76*
T0*
_output_shapes
:2
Identity_76�
AssignVariableOp_76AssignVariableOp+assignvariableop_76_adam_b_norm_3_13_beta_vIdentity_76:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_76_
Identity_77IdentityRestoreV2:tensors:77*
T0*
_output_shapes
:2
Identity_77�
AssignVariableOp_77AssignVariableOp*assignvariableop_77_adam_conv_4_5_kernel_vIdentity_77:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_77_
Identity_78IdentityRestoreV2:tensors:78*
T0*
_output_shapes
:2
Identity_78�
AssignVariableOp_78AssignVariableOp(assignvariableop_78_adam_conv_4_5_bias_vIdentity_78:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_78_
Identity_79IdentityRestoreV2:tensors:79*
T0*
_output_shapes
:2
Identity_79�
AssignVariableOp_79AssignVariableOp+assignvariableop_79_adam_b_norm_4_5_gamma_vIdentity_79:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_79_
Identity_80IdentityRestoreV2:tensors:80*
T0*
_output_shapes
:2
Identity_80�
AssignVariableOp_80AssignVariableOp*assignvariableop_80_adam_b_norm_4_5_beta_vIdentity_80:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_80_
Identity_81IdentityRestoreV2:tensors:81*
T0*
_output_shapes
:2
Identity_81�
AssignVariableOp_81AssignVariableOp>assignvariableop_81_adam_output_class_distribution_21_kernel_vIdentity_81:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_81_
Identity_82IdentityRestoreV2:tensors:82*
T0*
_output_shapes
:2
Identity_82�
AssignVariableOp_82AssignVariableOp<assignvariableop_82_adam_output_class_distribution_21_bias_vIdentity_82:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_82�
RestoreV2_1/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*1
value(B&B_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2_1/tensor_names�
RestoreV2_1/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*
valueB
B 2
RestoreV2_1/shape_and_slices�
RestoreV2_1	RestoreV2file_prefix!RestoreV2_1/tensor_names:output:0%RestoreV2_1/shape_and_slices:output:0
^RestoreV2"/device:CPU:0*
_output_shapes
:*
dtypes
22
RestoreV2_19
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOp�
Identity_83Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_55^AssignVariableOp_56^AssignVariableOp_57^AssignVariableOp_58^AssignVariableOp_59^AssignVariableOp_6^AssignVariableOp_60^AssignVariableOp_61^AssignVariableOp_62^AssignVariableOp_63^AssignVariableOp_64^AssignVariableOp_65^AssignVariableOp_66^AssignVariableOp_67^AssignVariableOp_68^AssignVariableOp_69^AssignVariableOp_7^AssignVariableOp_70^AssignVariableOp_71^AssignVariableOp_72^AssignVariableOp_73^AssignVariableOp_74^AssignVariableOp_75^AssignVariableOp_76^AssignVariableOp_77^AssignVariableOp_78^AssignVariableOp_79^AssignVariableOp_8^AssignVariableOp_80^AssignVariableOp_81^AssignVariableOp_82^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2
Identity_83�
Identity_84IdentityIdentity_83:output:0^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_55^AssignVariableOp_56^AssignVariableOp_57^AssignVariableOp_58^AssignVariableOp_59^AssignVariableOp_6^AssignVariableOp_60^AssignVariableOp_61^AssignVariableOp_62^AssignVariableOp_63^AssignVariableOp_64^AssignVariableOp_65^AssignVariableOp_66^AssignVariableOp_67^AssignVariableOp_68^AssignVariableOp_69^AssignVariableOp_7^AssignVariableOp_70^AssignVariableOp_71^AssignVariableOp_72^AssignVariableOp_73^AssignVariableOp_74^AssignVariableOp_75^AssignVariableOp_76^AssignVariableOp_77^AssignVariableOp_78^AssignVariableOp_79^AssignVariableOp_8^AssignVariableOp_80^AssignVariableOp_81^AssignVariableOp_82^AssignVariableOp_9
^RestoreV2^RestoreV2_1*
T0*
_output_shapes
: 2
Identity_84"#
identity_84Identity_84:output:0*�
_input_shapes�
�: :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
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
AssignVariableOp_2AssignVariableOp_22*
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
AssignVariableOp_3AssignVariableOp_32*
AssignVariableOp_30AssignVariableOp_302*
AssignVariableOp_31AssignVariableOp_312*
AssignVariableOp_32AssignVariableOp_322*
AssignVariableOp_33AssignVariableOp_332*
AssignVariableOp_34AssignVariableOp_342*
AssignVariableOp_35AssignVariableOp_352*
AssignVariableOp_36AssignVariableOp_362*
AssignVariableOp_37AssignVariableOp_372*
AssignVariableOp_38AssignVariableOp_382*
AssignVariableOp_39AssignVariableOp_392(
AssignVariableOp_4AssignVariableOp_42*
AssignVariableOp_40AssignVariableOp_402*
AssignVariableOp_41AssignVariableOp_412*
AssignVariableOp_42AssignVariableOp_422*
AssignVariableOp_43AssignVariableOp_432*
AssignVariableOp_44AssignVariableOp_442*
AssignVariableOp_45AssignVariableOp_452*
AssignVariableOp_46AssignVariableOp_462*
AssignVariableOp_47AssignVariableOp_472*
AssignVariableOp_48AssignVariableOp_482*
AssignVariableOp_49AssignVariableOp_492(
AssignVariableOp_5AssignVariableOp_52*
AssignVariableOp_50AssignVariableOp_502*
AssignVariableOp_51AssignVariableOp_512*
AssignVariableOp_52AssignVariableOp_522*
AssignVariableOp_53AssignVariableOp_532*
AssignVariableOp_54AssignVariableOp_542*
AssignVariableOp_55AssignVariableOp_552*
AssignVariableOp_56AssignVariableOp_562*
AssignVariableOp_57AssignVariableOp_572*
AssignVariableOp_58AssignVariableOp_582*
AssignVariableOp_59AssignVariableOp_592(
AssignVariableOp_6AssignVariableOp_62*
AssignVariableOp_60AssignVariableOp_602*
AssignVariableOp_61AssignVariableOp_612*
AssignVariableOp_62AssignVariableOp_622*
AssignVariableOp_63AssignVariableOp_632*
AssignVariableOp_64AssignVariableOp_642*
AssignVariableOp_65AssignVariableOp_652*
AssignVariableOp_66AssignVariableOp_662*
AssignVariableOp_67AssignVariableOp_672*
AssignVariableOp_68AssignVariableOp_682*
AssignVariableOp_69AssignVariableOp_692(
AssignVariableOp_7AssignVariableOp_72*
AssignVariableOp_70AssignVariableOp_702*
AssignVariableOp_71AssignVariableOp_712*
AssignVariableOp_72AssignVariableOp_722*
AssignVariableOp_73AssignVariableOp_732*
AssignVariableOp_74AssignVariableOp_742*
AssignVariableOp_75AssignVariableOp_752*
AssignVariableOp_76AssignVariableOp_762*
AssignVariableOp_77AssignVariableOp_772*
AssignVariableOp_78AssignVariableOp_782*
AssignVariableOp_79AssignVariableOp_792(
AssignVariableOp_8AssignVariableOp_82*
AssignVariableOp_80AssignVariableOp_802*
AssignVariableOp_81AssignVariableOp_812*
AssignVariableOp_82AssignVariableOp_822(
AssignVariableOp_9AssignVariableOp_92
	RestoreV2	RestoreV22
RestoreV2_1RestoreV2_1:+ '
%
_user_specified_namefile_prefix
�
�
;__inference_input_batch_normalization_layer_call_fn_4354348

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4*
Tin	
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*A
_output_shapes/
-:+���������������������������*-
config_proto

GPU

CPU2*0J 8*_
fZRX
V__inference_input_batch_normalization_layer_call_and_return_conditional_losses_43521772
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+���������������������������2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+���������������������������::::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs
�
�
V__inference_input_batch_normalization_layer_call_and_return_conditional_losses_4354330

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity��FusedBatchNormV3/ReadVariableOp�!FusedBatchNormV3/ReadVariableOp_1�ReadVariableOp�ReadVariableOp_1^
LogicalAnd/xConst*
_output_shapes
: *
dtype0
*
value	B
 Z 2
LogicalAnd/x^
LogicalAnd/yConst*
_output_shapes
: *
dtype0
*
value	B
 Z2
LogicalAnd/yl

LogicalAnd
LogicalAndLogicalAnd/x:output:0LogicalAnd/y:output:0*
_output_shapes
: 2

LogicalAndt
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOp�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+���������������������������:::::*
epsilon%o�:*
is_training( 2
FusedBatchNormV3S
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *�p}?2
Const�
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*A
_output_shapes/
-:+���������������������������2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+���������������������������::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:& "
 
_user_specified_nameinputs
�
_
C__inference_RELU_2_layer_call_and_return_conditional_losses_4354802

inputs
identityV
ReluReluinputs*
T0*/
_output_shapes
:���������   2
Relun
IdentityIdentityRelu:activations:0*
T0*/
_output_shapes
:���������   2

Identity"
identityIdentity:output:0*.
_input_shapes
:���������   :& "
 
_user_specified_nameinputs
�
�
(__inference_conv_4_layer_call_fn_4352708

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*A
_output_shapes/
-:+���������������������������@*-
config_proto

GPU

CPU2*0J 8*L
fGRE
C__inference_conv_4_layer_call_and_return_conditional_losses_43527002
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+���������������������������@2

Identity"
identityIdentity:output:0*H
_input_shapes7
5:+���������������������������@::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs
�$
�
E__inference_b_norm_4_layer_call_and_return_conditional_losses_4352802

inputs
readvariableop_resource
readvariableop_1_resource
assignmovingavg_4352787
assignmovingavg_1_4352794
identity��#AssignMovingAvg/AssignSubVariableOp�AssignMovingAvg/ReadVariableOp�%AssignMovingAvg_1/AssignSubVariableOp� AssignMovingAvg_1/ReadVariableOp�ReadVariableOp�ReadVariableOp_1^
LogicalAnd/xConst*
_output_shapes
: *
dtype0
*
value	B
 Z2
LogicalAnd/x^
LogicalAnd/yConst*
_output_shapes
: *
dtype0
*
value	B
 Z2
LogicalAnd/yl

LogicalAnd
LogicalAndLogicalAnd/x:output:0LogicalAnd/y:output:0*
_output_shapes
: 2

LogicalAndt
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1Q
ConstConst*
_output_shapes
: *
dtype0*
valueB 2
ConstU
Const_1Const*
_output_shapes
: *
dtype0*
valueB 2	
Const_1�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0Const:output:0Const_1:output:0*
T0*
U0*]
_output_shapesK
I:+���������������������������@:@:@:@:@:*
epsilon%o�:2
FusedBatchNormV3W
Const_2Const*
_output_shapes
: *
dtype0*
valueB
 *�p}?2	
Const_2�
AssignMovingAvg/sub/xConst**
_class 
loc:@AssignMovingAvg/4352787*
_output_shapes
: *
dtype0*
valueB
 *  �?2
AssignMovingAvg/sub/x�
AssignMovingAvg/subSubAssignMovingAvg/sub/x:output:0Const_2:output:0*
T0**
_class 
loc:@AssignMovingAvg/4352787*
_output_shapes
: 2
AssignMovingAvg/sub�
AssignMovingAvg/ReadVariableOpReadVariableOpassignmovingavg_4352787*
_output_shapes
:@*
dtype02 
AssignMovingAvg/ReadVariableOp�
AssignMovingAvg/sub_1Sub&AssignMovingAvg/ReadVariableOp:value:0FusedBatchNormV3:batch_mean:0*
T0**
_class 
loc:@AssignMovingAvg/4352787*
_output_shapes
:@2
AssignMovingAvg/sub_1�
AssignMovingAvg/mulMulAssignMovingAvg/sub_1:z:0AssignMovingAvg/sub:z:0*
T0**
_class 
loc:@AssignMovingAvg/4352787*
_output_shapes
:@2
AssignMovingAvg/mul�
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOpassignmovingavg_4352787AssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp**
_class 
loc:@AssignMovingAvg/4352787*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOp�
AssignMovingAvg_1/sub/xConst*,
_class"
 loc:@AssignMovingAvg_1/4352794*
_output_shapes
: *
dtype0*
valueB
 *  �?2
AssignMovingAvg_1/sub/x�
AssignMovingAvg_1/subSub AssignMovingAvg_1/sub/x:output:0Const_2:output:0*
T0*,
_class"
 loc:@AssignMovingAvg_1/4352794*
_output_shapes
: 2
AssignMovingAvg_1/sub�
 AssignMovingAvg_1/ReadVariableOpReadVariableOpassignmovingavg_1_4352794*
_output_shapes
:@*
dtype02"
 AssignMovingAvg_1/ReadVariableOp�
AssignMovingAvg_1/sub_1Sub(AssignMovingAvg_1/ReadVariableOp:value:0!FusedBatchNormV3:batch_variance:0*
T0*,
_class"
 loc:@AssignMovingAvg_1/4352794*
_output_shapes
:@2
AssignMovingAvg_1/sub_1�
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub_1:z:0AssignMovingAvg_1/sub:z:0*
T0*,
_class"
 loc:@AssignMovingAvg_1/4352794*
_output_shapes
:@2
AssignMovingAvg_1/mul�
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOpassignmovingavg_1_4352794AssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*,
_class"
 loc:@AssignMovingAvg_1/4352794*
_output_shapes
 *
dtype02'
%AssignMovingAvg_1/AssignSubVariableOp�
IdentityIdentityFusedBatchNormV3:y:0$^AssignMovingAvg/AssignSubVariableOp^AssignMovingAvg/ReadVariableOp&^AssignMovingAvg_1/AssignSubVariableOp!^AssignMovingAvg_1/ReadVariableOp^ReadVariableOp^ReadVariableOp_1*
T0*A
_output_shapes/
-:+���������������������������@2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+���������������������������@::::2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp2D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp2 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:& "
 
_user_specified_nameinputs
�
H
,__inference_dropout_43_layer_call_fn_4355193

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*(
_output_shapes
:����������*-
config_proto

GPU

CPU2*0J 8*P
fKRI
G__inference_dropout_43_layer_call_and_return_conditional_losses_43534132
PartitionedCallm
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:����������2

Identity"
identityIdentity:output:0*'
_input_shapes
:����������:& "
 
_user_specified_nameinputs
�
_
C__inference_RELU_1_layer_call_and_return_conditional_losses_4354632

inputs
identityV
ReluReluinputs*
T0*/
_output_shapes
:���������@@2
Relun
IdentityIdentityRelu:activations:0*
T0*/
_output_shapes
:���������@@2

Identity"
identityIdentity:output:0*.
_input_shapes
:���������@@:& "
 
_user_specified_nameinputs
�$
�
E__inference_b_norm_2_layer_call_and_return_conditional_losses_4352474

inputs
readvariableop_resource
readvariableop_1_resource
assignmovingavg_4352459
assignmovingavg_1_4352466
identity��#AssignMovingAvg/AssignSubVariableOp�AssignMovingAvg/ReadVariableOp�%AssignMovingAvg_1/AssignSubVariableOp� AssignMovingAvg_1/ReadVariableOp�ReadVariableOp�ReadVariableOp_1^
LogicalAnd/xConst*
_output_shapes
: *
dtype0
*
value	B
 Z2
LogicalAnd/x^
LogicalAnd/yConst*
_output_shapes
: *
dtype0
*
value	B
 Z2
LogicalAnd/yl

LogicalAnd
LogicalAndLogicalAnd/x:output:0LogicalAnd/y:output:0*
_output_shapes
: 2

LogicalAndt
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype02
ReadVariableOp_1Q
ConstConst*
_output_shapes
: *
dtype0*
valueB 2
ConstU
Const_1Const*
_output_shapes
: *
dtype0*
valueB 2	
Const_1�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0Const:output:0Const_1:output:0*
T0*
U0*]
_output_shapesK
I:+��������������������������� : : : : :*
epsilon%o�:2
FusedBatchNormV3W
Const_2Const*
_output_shapes
: *
dtype0*
valueB
 *�p}?2	
Const_2�
AssignMovingAvg/sub/xConst**
_class 
loc:@AssignMovingAvg/4352459*
_output_shapes
: *
dtype0*
valueB
 *  �?2
AssignMovingAvg/sub/x�
AssignMovingAvg/subSubAssignMovingAvg/sub/x:output:0Const_2:output:0*
T0**
_class 
loc:@AssignMovingAvg/4352459*
_output_shapes
: 2
AssignMovingAvg/sub�
AssignMovingAvg/ReadVariableOpReadVariableOpassignmovingavg_4352459*
_output_shapes
: *
dtype02 
AssignMovingAvg/ReadVariableOp�
AssignMovingAvg/sub_1Sub&AssignMovingAvg/ReadVariableOp:value:0FusedBatchNormV3:batch_mean:0*
T0**
_class 
loc:@AssignMovingAvg/4352459*
_output_shapes
: 2
AssignMovingAvg/sub_1�
AssignMovingAvg/mulMulAssignMovingAvg/sub_1:z:0AssignMovingAvg/sub:z:0*
T0**
_class 
loc:@AssignMovingAvg/4352459*
_output_shapes
: 2
AssignMovingAvg/mul�
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOpassignmovingavg_4352459AssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp**
_class 
loc:@AssignMovingAvg/4352459*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOp�
AssignMovingAvg_1/sub/xConst*,
_class"
 loc:@AssignMovingAvg_1/4352466*
_output_shapes
: *
dtype0*
valueB
 *  �?2
AssignMovingAvg_1/sub/x�
AssignMovingAvg_1/subSub AssignMovingAvg_1/sub/x:output:0Const_2:output:0*
T0*,
_class"
 loc:@AssignMovingAvg_1/4352466*
_output_shapes
: 2
AssignMovingAvg_1/sub�
 AssignMovingAvg_1/ReadVariableOpReadVariableOpassignmovingavg_1_4352466*
_output_shapes
: *
dtype02"
 AssignMovingAvg_1/ReadVariableOp�
AssignMovingAvg_1/sub_1Sub(AssignMovingAvg_1/ReadVariableOp:value:0!FusedBatchNormV3:batch_variance:0*
T0*,
_class"
 loc:@AssignMovingAvg_1/4352466*
_output_shapes
: 2
AssignMovingAvg_1/sub_1�
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub_1:z:0AssignMovingAvg_1/sub:z:0*
T0*,
_class"
 loc:@AssignMovingAvg_1/4352466*
_output_shapes
: 2
AssignMovingAvg_1/mul�
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOpassignmovingavg_1_4352466AssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*,
_class"
 loc:@AssignMovingAvg_1/4352466*
_output_shapes
 *
dtype02'
%AssignMovingAvg_1/AssignSubVariableOp�
IdentityIdentityFusedBatchNormV3:y:0$^AssignMovingAvg/AssignSubVariableOp^AssignMovingAvg/ReadVariableOp&^AssignMovingAvg_1/AssignSubVariableOp!^AssignMovingAvg_1/ReadVariableOp^ReadVariableOp^ReadVariableOp_1*
T0*A
_output_shapes/
-:+��������������������������� 2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+��������������������������� ::::2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp2D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp2 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:& "
 
_user_specified_nameinputs
�
i
M__inference_max_pooling2d_84_layer_call_and_return_conditional_losses_4352846

inputs
identity�
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4������������������������������������*
ksize
*
paddingVALID*
strides
2	
MaxPool�
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4������������������������������������2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4������������������������������������:& "
 
_user_specified_nameinputs
�
c
G__inference_flatten_21_layer_call_and_return_conditional_losses_4355153

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"����   2
Consth
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:����������2	
Reshapee
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:����������2

Identity"
identityIdentity:output:0*.
_input_shapes
:���������@:& "
 
_user_specified_nameinputs
�
�
E__inference_b_norm_2_layer_call_and_return_conditional_losses_4352505

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity��FusedBatchNormV3/ReadVariableOp�!FusedBatchNormV3/ReadVariableOp_1�ReadVariableOp�ReadVariableOp_1^
LogicalAnd/xConst*
_output_shapes
: *
dtype0
*
value	B
 Z 2
LogicalAnd/x^
LogicalAnd/yConst*
_output_shapes
: *
dtype0
*
value	B
 Z2
LogicalAnd/yl

LogicalAnd
LogicalAndLogicalAnd/x:output:0LogicalAnd/y:output:0*
_output_shapes
: 2

LogicalAndt
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype02
ReadVariableOp_1�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02!
FusedBatchNormV3/ReadVariableOp�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02#
!FusedBatchNormV3/ReadVariableOp_1�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+��������������������������� : : : : :*
epsilon%o�:*
is_training( 2
FusedBatchNormV3S
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *�p}?2
Const�
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*A
_output_shapes/
-:+��������������������������� 2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+��������������������������� ::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:& "
 
_user_specified_nameinputs
�
f
G__inference_dropout_43_layer_call_and_return_conditional_losses_4353408

inputs
identity�a
dropout/rateConst*
_output_shapes
: *
dtype0*
valueB
 *33�>2
dropout/rateT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape}
dropout/random_uniform/minConst*
_output_shapes
: *
dtype0*
valueB
 *    2
dropout/random_uniform/min}
dropout/random_uniform/maxConst*
_output_shapes
: *
dtype0*
valueB
 *  �?2
dropout/random_uniform/max�
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*(
_output_shapes
:����������*
dtype02&
$dropout/random_uniform/RandomUniform�
dropout/random_uniform/subSub#dropout/random_uniform/max:output:0#dropout/random_uniform/min:output:0*
T0*
_output_shapes
: 2
dropout/random_uniform/sub�
dropout/random_uniform/mulMul-dropout/random_uniform/RandomUniform:output:0dropout/random_uniform/sub:z:0*
T0*(
_output_shapes
:����������2
dropout/random_uniform/mul�
dropout/random_uniformAdddropout/random_uniform/mul:z:0#dropout/random_uniform/min:output:0*
T0*(
_output_shapes
:����������2
dropout/random_uniformc
dropout/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?2
dropout/sub/xq
dropout/subSubdropout/sub/x:output:0dropout/rate:output:0*
T0*
_output_shapes
: 2
dropout/subk
dropout/truediv/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?2
dropout/truediv/x{
dropout/truedivRealDivdropout/truediv/x:output:0dropout/sub:z:0*
T0*
_output_shapes
: 2
dropout/truediv�
dropout/GreaterEqualGreaterEqualdropout/random_uniform:z:0dropout/rate:output:0*
T0*(
_output_shapes
:����������2
dropout/GreaterEqualq
dropout/mulMulinputsdropout/truediv:z:0*
T0*(
_output_shapes
:����������2
dropout/mul�
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:����������2
dropout/Cast{
dropout/mul_1Muldropout/mul:z:0dropout/Cast:y:0*
T0*(
_output_shapes
:����������2
dropout/mul_1f
IdentityIdentitydropout/mul_1:z:0*
T0*(
_output_shapes
:����������2

Identity"
identityIdentity:output:0*'
_input_shapes
:����������:& "
 
_user_specified_nameinputs
�
�
(__inference_conv_3_layer_call_fn_4352544

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*A
_output_shapes/
-:+���������������������������@*-
config_proto

GPU

CPU2*0J 8*L
fGRE
C__inference_conv_3_layer_call_and_return_conditional_losses_43525362
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+���������������������������@2

Identity"
identityIdentity:output:0*H
_input_shapes7
5:+��������������������������� ::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs
�
e
,__inference_dropout_43_layer_call_fn_4355188

inputs
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*(
_output_shapes
:����������*-
config_proto

GPU

CPU2*0J 8*P
fKRI
G__inference_dropout_43_layer_call_and_return_conditional_losses_43534082
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:����������2

Identity"
identityIdentity:output:0*'
_input_shapes
:����������22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs
��
�
G__inference_awsome_net_layer_call_and_return_conditional_losses_4354188

inputs-
)input_conv_conv2d_readvariableop_resource.
*input_conv_biasadd_readvariableop_resource5
1input_batch_normalization_readvariableop_resource7
3input_batch_normalization_readvariableop_1_resourceF
Binput_batch_normalization_fusedbatchnormv3_readvariableop_resourceH
Dinput_batch_normalization_fusedbatchnormv3_readvariableop_1_resource)
%conv_1_conv2d_readvariableop_resource*
&conv_1_biasadd_readvariableop_resource$
 b_norm_1_readvariableop_resource&
"b_norm_1_readvariableop_1_resource5
1b_norm_1_fusedbatchnormv3_readvariableop_resource7
3b_norm_1_fusedbatchnormv3_readvariableop_1_resource)
%conv_2_conv2d_readvariableop_resource*
&conv_2_biasadd_readvariableop_resource$
 b_norm_2_readvariableop_resource&
"b_norm_2_readvariableop_1_resource5
1b_norm_2_fusedbatchnormv3_readvariableop_resource7
3b_norm_2_fusedbatchnormv3_readvariableop_1_resource)
%conv_3_conv2d_readvariableop_resource*
&conv_3_biasadd_readvariableop_resource$
 b_norm_3_readvariableop_resource&
"b_norm_3_readvariableop_1_resource5
1b_norm_3_fusedbatchnormv3_readvariableop_resource7
3b_norm_3_fusedbatchnormv3_readvariableop_1_resource)
%conv_4_conv2d_readvariableop_resource*
&conv_4_biasadd_readvariableop_resource$
 b_norm_4_readvariableop_resource&
"b_norm_4_readvariableop_1_resource5
1b_norm_4_fusedbatchnormv3_readvariableop_resource7
3b_norm_4_fusedbatchnormv3_readvariableop_1_resource<
8output_class_distribution_matmul_readvariableop_resource=
9output_class_distribution_biasadd_readvariableop_resource
identity��(b_norm_1/FusedBatchNormV3/ReadVariableOp�*b_norm_1/FusedBatchNormV3/ReadVariableOp_1�b_norm_1/ReadVariableOp�b_norm_1/ReadVariableOp_1�(b_norm_2/FusedBatchNormV3/ReadVariableOp�*b_norm_2/FusedBatchNormV3/ReadVariableOp_1�b_norm_2/ReadVariableOp�b_norm_2/ReadVariableOp_1�(b_norm_3/FusedBatchNormV3/ReadVariableOp�*b_norm_3/FusedBatchNormV3/ReadVariableOp_1�b_norm_3/ReadVariableOp�b_norm_3/ReadVariableOp_1�(b_norm_4/FusedBatchNormV3/ReadVariableOp�*b_norm_4/FusedBatchNormV3/ReadVariableOp_1�b_norm_4/ReadVariableOp�b_norm_4/ReadVariableOp_1�conv_1/BiasAdd/ReadVariableOp�conv_1/Conv2D/ReadVariableOp�conv_2/BiasAdd/ReadVariableOp�conv_2/Conv2D/ReadVariableOp�conv_3/BiasAdd/ReadVariableOp�conv_3/Conv2D/ReadVariableOp�conv_4/BiasAdd/ReadVariableOp�conv_4/Conv2D/ReadVariableOp�9input_batch_normalization/FusedBatchNormV3/ReadVariableOp�;input_batch_normalization/FusedBatchNormV3/ReadVariableOp_1�(input_batch_normalization/ReadVariableOp�*input_batch_normalization/ReadVariableOp_1�!input_conv/BiasAdd/ReadVariableOp� input_conv/Conv2D/ReadVariableOp�0output_class_distribution/BiasAdd/ReadVariableOp�/output_class_distribution/MatMul/ReadVariableOp�
 input_conv/Conv2D/ReadVariableOpReadVariableOp)input_conv_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02"
 input_conv/Conv2D/ReadVariableOp�
input_conv/Conv2DConv2Dinputs(input_conv/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:�����������*
paddingSAME*
strides
2
input_conv/Conv2D�
!input_conv/BiasAdd/ReadVariableOpReadVariableOp*input_conv_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02#
!input_conv/BiasAdd/ReadVariableOp�
input_conv/BiasAddBiasAddinput_conv/Conv2D:output:0)input_conv/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:�����������2
input_conv/BiasAdd�
&input_batch_normalization/LogicalAnd/xConst*
_output_shapes
: *
dtype0
*
value	B
 Z 2(
&input_batch_normalization/LogicalAnd/x�
&input_batch_normalization/LogicalAnd/yConst*
_output_shapes
: *
dtype0
*
value	B
 Z2(
&input_batch_normalization/LogicalAnd/y�
$input_batch_normalization/LogicalAnd
LogicalAnd/input_batch_normalization/LogicalAnd/x:output:0/input_batch_normalization/LogicalAnd/y:output:0*
_output_shapes
: 2&
$input_batch_normalization/LogicalAnd�
(input_batch_normalization/ReadVariableOpReadVariableOp1input_batch_normalization_readvariableop_resource*
_output_shapes
:*
dtype02*
(input_batch_normalization/ReadVariableOp�
*input_batch_normalization/ReadVariableOp_1ReadVariableOp3input_batch_normalization_readvariableop_1_resource*
_output_shapes
:*
dtype02,
*input_batch_normalization/ReadVariableOp_1�
9input_batch_normalization/FusedBatchNormV3/ReadVariableOpReadVariableOpBinput_batch_normalization_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02;
9input_batch_normalization/FusedBatchNormV3/ReadVariableOp�
;input_batch_normalization/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpDinput_batch_normalization_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02=
;input_batch_normalization/FusedBatchNormV3/ReadVariableOp_1�
*input_batch_normalization/FusedBatchNormV3FusedBatchNormV3input_conv/BiasAdd:output:00input_batch_normalization/ReadVariableOp:value:02input_batch_normalization/ReadVariableOp_1:value:0Ainput_batch_normalization/FusedBatchNormV3/ReadVariableOp:value:0Cinput_batch_normalization/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*M
_output_shapes;
9:�����������:::::*
epsilon%o�:*
is_training( 2,
*input_batch_normalization/FusedBatchNormV3�
input_batch_normalization/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *�p}?2!
input_batch_normalization/Const�
input_RELU/ReluRelu.input_batch_normalization/FusedBatchNormV3:y:0*
T0*1
_output_shapes
:�����������2
input_RELU/Relu�
max_pooling2d_85/MaxPoolMaxPoolinput_RELU/Relu:activations:0*/
_output_shapes
:���������@@*
ksize
*
paddingVALID*
strides
2
max_pooling2d_85/MaxPool�
dropout_42/IdentityIdentity!max_pooling2d_85/MaxPool:output:0*
T0*/
_output_shapes
:���������@@2
dropout_42/Identity�
conv_1/Conv2D/ReadVariableOpReadVariableOp%conv_1_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02
conv_1/Conv2D/ReadVariableOp�
conv_1/Conv2DConv2Ddropout_42/Identity:output:0$conv_1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@@*
paddingSAME*
strides
2
conv_1/Conv2D�
conv_1/BiasAdd/ReadVariableOpReadVariableOp&conv_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02
conv_1/BiasAdd/ReadVariableOp�
conv_1/BiasAddBiasAddconv_1/Conv2D:output:0%conv_1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@@2
conv_1/BiasAddp
b_norm_1/LogicalAnd/xConst*
_output_shapes
: *
dtype0
*
value	B
 Z 2
b_norm_1/LogicalAnd/xp
b_norm_1/LogicalAnd/yConst*
_output_shapes
: *
dtype0
*
value	B
 Z2
b_norm_1/LogicalAnd/y�
b_norm_1/LogicalAnd
LogicalAndb_norm_1/LogicalAnd/x:output:0b_norm_1/LogicalAnd/y:output:0*
_output_shapes
: 2
b_norm_1/LogicalAnd�
b_norm_1/ReadVariableOpReadVariableOp b_norm_1_readvariableop_resource*
_output_shapes
:*
dtype02
b_norm_1/ReadVariableOp�
b_norm_1/ReadVariableOp_1ReadVariableOp"b_norm_1_readvariableop_1_resource*
_output_shapes
:*
dtype02
b_norm_1/ReadVariableOp_1�
(b_norm_1/FusedBatchNormV3/ReadVariableOpReadVariableOp1b_norm_1_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02*
(b_norm_1/FusedBatchNormV3/ReadVariableOp�
*b_norm_1/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp3b_norm_1_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02,
*b_norm_1/FusedBatchNormV3/ReadVariableOp_1�
b_norm_1/FusedBatchNormV3FusedBatchNormV3conv_1/BiasAdd:output:0b_norm_1/ReadVariableOp:value:0!b_norm_1/ReadVariableOp_1:value:00b_norm_1/FusedBatchNormV3/ReadVariableOp:value:02b_norm_1/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:���������@@:::::*
epsilon%o�:*
is_training( 2
b_norm_1/FusedBatchNormV3e
b_norm_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *�p}?2
b_norm_1/Const{
RELU_1/ReluRelub_norm_1/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:���������@@2
RELU_1/Relu�
max_pooling2d_81/MaxPoolMaxPoolRELU_1/Relu:activations:0*/
_output_shapes
:���������  *
ksize
*
paddingVALID*
strides
2
max_pooling2d_81/MaxPool�
conv_2/Conv2D/ReadVariableOpReadVariableOp%conv_2_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02
conv_2/Conv2D/ReadVariableOp�
conv_2/Conv2DConv2D!max_pooling2d_81/MaxPool:output:0$conv_2/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������   *
paddingSAME*
strides
2
conv_2/Conv2D�
conv_2/BiasAdd/ReadVariableOpReadVariableOp&conv_2_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02
conv_2/BiasAdd/ReadVariableOp�
conv_2/BiasAddBiasAddconv_2/Conv2D:output:0%conv_2/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������   2
conv_2/BiasAddp
b_norm_2/LogicalAnd/xConst*
_output_shapes
: *
dtype0
*
value	B
 Z 2
b_norm_2/LogicalAnd/xp
b_norm_2/LogicalAnd/yConst*
_output_shapes
: *
dtype0
*
value	B
 Z2
b_norm_2/LogicalAnd/y�
b_norm_2/LogicalAnd
LogicalAndb_norm_2/LogicalAnd/x:output:0b_norm_2/LogicalAnd/y:output:0*
_output_shapes
: 2
b_norm_2/LogicalAnd�
b_norm_2/ReadVariableOpReadVariableOp b_norm_2_readvariableop_resource*
_output_shapes
: *
dtype02
b_norm_2/ReadVariableOp�
b_norm_2/ReadVariableOp_1ReadVariableOp"b_norm_2_readvariableop_1_resource*
_output_shapes
: *
dtype02
b_norm_2/ReadVariableOp_1�
(b_norm_2/FusedBatchNormV3/ReadVariableOpReadVariableOp1b_norm_2_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02*
(b_norm_2/FusedBatchNormV3/ReadVariableOp�
*b_norm_2/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp3b_norm_2_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02,
*b_norm_2/FusedBatchNormV3/ReadVariableOp_1�
b_norm_2/FusedBatchNormV3FusedBatchNormV3conv_2/BiasAdd:output:0b_norm_2/ReadVariableOp:value:0!b_norm_2/ReadVariableOp_1:value:00b_norm_2/FusedBatchNormV3/ReadVariableOp:value:02b_norm_2/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:���������   : : : : :*
epsilon%o�:*
is_training( 2
b_norm_2/FusedBatchNormV3e
b_norm_2/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *�p}?2
b_norm_2/Const{
RELU_2/ReluRelub_norm_2/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:���������   2
RELU_2/Relu�
max_pooling2d_82/MaxPoolMaxPoolRELU_2/Relu:activations:0*/
_output_shapes
:��������� *
ksize
*
paddingVALID*
strides
2
max_pooling2d_82/MaxPool�
conv_3/Conv2D/ReadVariableOpReadVariableOp%conv_3_conv2d_readvariableop_resource*&
_output_shapes
: @*
dtype02
conv_3/Conv2D/ReadVariableOp�
conv_3/Conv2DConv2D!max_pooling2d_82/MaxPool:output:0$conv_3/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@*
paddingSAME*
strides
2
conv_3/Conv2D�
conv_3/BiasAdd/ReadVariableOpReadVariableOp&conv_3_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
conv_3/BiasAdd/ReadVariableOp�
conv_3/BiasAddBiasAddconv_3/Conv2D:output:0%conv_3/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@2
conv_3/BiasAddp
b_norm_3/LogicalAnd/xConst*
_output_shapes
: *
dtype0
*
value	B
 Z 2
b_norm_3/LogicalAnd/xp
b_norm_3/LogicalAnd/yConst*
_output_shapes
: *
dtype0
*
value	B
 Z2
b_norm_3/LogicalAnd/y�
b_norm_3/LogicalAnd
LogicalAndb_norm_3/LogicalAnd/x:output:0b_norm_3/LogicalAnd/y:output:0*
_output_shapes
: 2
b_norm_3/LogicalAnd�
b_norm_3/ReadVariableOpReadVariableOp b_norm_3_readvariableop_resource*
_output_shapes
:@*
dtype02
b_norm_3/ReadVariableOp�
b_norm_3/ReadVariableOp_1ReadVariableOp"b_norm_3_readvariableop_1_resource*
_output_shapes
:@*
dtype02
b_norm_3/ReadVariableOp_1�
(b_norm_3/FusedBatchNormV3/ReadVariableOpReadVariableOp1b_norm_3_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02*
(b_norm_3/FusedBatchNormV3/ReadVariableOp�
*b_norm_3/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp3b_norm_3_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02,
*b_norm_3/FusedBatchNormV3/ReadVariableOp_1�
b_norm_3/FusedBatchNormV3FusedBatchNormV3conv_3/BiasAdd:output:0b_norm_3/ReadVariableOp:value:0!b_norm_3/ReadVariableOp_1:value:00b_norm_3/FusedBatchNormV3/ReadVariableOp:value:02b_norm_3/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:���������@:@:@:@:@:*
epsilon%o�:*
is_training( 2
b_norm_3/FusedBatchNormV3e
b_norm_3/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *�p}?2
b_norm_3/Const{
RELU_3/ReluRelub_norm_3/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:���������@2
RELU_3/Relu�
max_pooling2d_83/MaxPoolMaxPoolRELU_3/Relu:activations:0*/
_output_shapes
:���������@*
ksize
*
paddingVALID*
strides
2
max_pooling2d_83/MaxPool�
conv_4/Conv2D/ReadVariableOpReadVariableOp%conv_4_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02
conv_4/Conv2D/ReadVariableOp�
conv_4/Conv2DConv2D!max_pooling2d_83/MaxPool:output:0$conv_4/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@*
paddingSAME*
strides
2
conv_4/Conv2D�
conv_4/BiasAdd/ReadVariableOpReadVariableOp&conv_4_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
conv_4/BiasAdd/ReadVariableOp�
conv_4/BiasAddBiasAddconv_4/Conv2D:output:0%conv_4/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@2
conv_4/BiasAddp
b_norm_4/LogicalAnd/xConst*
_output_shapes
: *
dtype0
*
value	B
 Z 2
b_norm_4/LogicalAnd/xp
b_norm_4/LogicalAnd/yConst*
_output_shapes
: *
dtype0
*
value	B
 Z2
b_norm_4/LogicalAnd/y�
b_norm_4/LogicalAnd
LogicalAndb_norm_4/LogicalAnd/x:output:0b_norm_4/LogicalAnd/y:output:0*
_output_shapes
: 2
b_norm_4/LogicalAnd�
b_norm_4/ReadVariableOpReadVariableOp b_norm_4_readvariableop_resource*
_output_shapes
:@*
dtype02
b_norm_4/ReadVariableOp�
b_norm_4/ReadVariableOp_1ReadVariableOp"b_norm_4_readvariableop_1_resource*
_output_shapes
:@*
dtype02
b_norm_4/ReadVariableOp_1�
(b_norm_4/FusedBatchNormV3/ReadVariableOpReadVariableOp1b_norm_4_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02*
(b_norm_4/FusedBatchNormV3/ReadVariableOp�
*b_norm_4/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp3b_norm_4_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02,
*b_norm_4/FusedBatchNormV3/ReadVariableOp_1�
b_norm_4/FusedBatchNormV3FusedBatchNormV3conv_4/BiasAdd:output:0b_norm_4/ReadVariableOp:value:0!b_norm_4/ReadVariableOp_1:value:00b_norm_4/FusedBatchNormV3/ReadVariableOp:value:02b_norm_4/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:���������@:@:@:@:@:*
epsilon%o�:*
is_training( 2
b_norm_4/FusedBatchNormV3e
b_norm_4/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *�p}?2
b_norm_4/Const{
RELU_4/ReluRelub_norm_4/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:���������@2
RELU_4/Relu�
max_pooling2d_84/MaxPoolMaxPoolRELU_4/Relu:activations:0*/
_output_shapes
:���������@*
ksize
*
paddingVALID*
strides
2
max_pooling2d_84/MaxPoolu
flatten_21/ConstConst*
_output_shapes
:*
dtype0*
valueB"����   2
flatten_21/Const�
flatten_21/ReshapeReshape!max_pooling2d_84/MaxPool:output:0flatten_21/Const:output:0*
T0*(
_output_shapes
:����������2
flatten_21/Reshape�
dropout_43/IdentityIdentityflatten_21/Reshape:output:0*
T0*(
_output_shapes
:����������2
dropout_43/Identity�
/output_class_distribution/MatMul/ReadVariableOpReadVariableOp8output_class_distribution_matmul_readvariableop_resource* 
_output_shapes
:
��*
dtype021
/output_class_distribution/MatMul/ReadVariableOp�
 output_class_distribution/MatMulMatMuldropout_43/Identity:output:07output_class_distribution/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2"
 output_class_distribution/MatMul�
0output_class_distribution/BiasAdd/ReadVariableOpReadVariableOp9output_class_distribution_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype022
0output_class_distribution/BiasAdd/ReadVariableOp�
!output_class_distribution/BiasAddBiasAdd*output_class_distribution/MatMul:product:08output_class_distribution/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2#
!output_class_distribution/BiasAdd�

IdentityIdentity*output_class_distribution/BiasAdd:output:0)^b_norm_1/FusedBatchNormV3/ReadVariableOp+^b_norm_1/FusedBatchNormV3/ReadVariableOp_1^b_norm_1/ReadVariableOp^b_norm_1/ReadVariableOp_1)^b_norm_2/FusedBatchNormV3/ReadVariableOp+^b_norm_2/FusedBatchNormV3/ReadVariableOp_1^b_norm_2/ReadVariableOp^b_norm_2/ReadVariableOp_1)^b_norm_3/FusedBatchNormV3/ReadVariableOp+^b_norm_3/FusedBatchNormV3/ReadVariableOp_1^b_norm_3/ReadVariableOp^b_norm_3/ReadVariableOp_1)^b_norm_4/FusedBatchNormV3/ReadVariableOp+^b_norm_4/FusedBatchNormV3/ReadVariableOp_1^b_norm_4/ReadVariableOp^b_norm_4/ReadVariableOp_1^conv_1/BiasAdd/ReadVariableOp^conv_1/Conv2D/ReadVariableOp^conv_2/BiasAdd/ReadVariableOp^conv_2/Conv2D/ReadVariableOp^conv_3/BiasAdd/ReadVariableOp^conv_3/Conv2D/ReadVariableOp^conv_4/BiasAdd/ReadVariableOp^conv_4/Conv2D/ReadVariableOp:^input_batch_normalization/FusedBatchNormV3/ReadVariableOp<^input_batch_normalization/FusedBatchNormV3/ReadVariableOp_1)^input_batch_normalization/ReadVariableOp+^input_batch_normalization/ReadVariableOp_1"^input_conv/BiasAdd/ReadVariableOp!^input_conv/Conv2D/ReadVariableOp1^output_class_distribution/BiasAdd/ReadVariableOp0^output_class_distribution/MatMul/ReadVariableOp*
T0*(
_output_shapes
:����������2

Identity"
identityIdentity:output:0*�
_input_shapes�
�:�����������::::::::::::::::::::::::::::::::2T
(b_norm_1/FusedBatchNormV3/ReadVariableOp(b_norm_1/FusedBatchNormV3/ReadVariableOp2X
*b_norm_1/FusedBatchNormV3/ReadVariableOp_1*b_norm_1/FusedBatchNormV3/ReadVariableOp_122
b_norm_1/ReadVariableOpb_norm_1/ReadVariableOp26
b_norm_1/ReadVariableOp_1b_norm_1/ReadVariableOp_12T
(b_norm_2/FusedBatchNormV3/ReadVariableOp(b_norm_2/FusedBatchNormV3/ReadVariableOp2X
*b_norm_2/FusedBatchNormV3/ReadVariableOp_1*b_norm_2/FusedBatchNormV3/ReadVariableOp_122
b_norm_2/ReadVariableOpb_norm_2/ReadVariableOp26
b_norm_2/ReadVariableOp_1b_norm_2/ReadVariableOp_12T
(b_norm_3/FusedBatchNormV3/ReadVariableOp(b_norm_3/FusedBatchNormV3/ReadVariableOp2X
*b_norm_3/FusedBatchNormV3/ReadVariableOp_1*b_norm_3/FusedBatchNormV3/ReadVariableOp_122
b_norm_3/ReadVariableOpb_norm_3/ReadVariableOp26
b_norm_3/ReadVariableOp_1b_norm_3/ReadVariableOp_12T
(b_norm_4/FusedBatchNormV3/ReadVariableOp(b_norm_4/FusedBatchNormV3/ReadVariableOp2X
*b_norm_4/FusedBatchNormV3/ReadVariableOp_1*b_norm_4/FusedBatchNormV3/ReadVariableOp_122
b_norm_4/ReadVariableOpb_norm_4/ReadVariableOp26
b_norm_4/ReadVariableOp_1b_norm_4/ReadVariableOp_12>
conv_1/BiasAdd/ReadVariableOpconv_1/BiasAdd/ReadVariableOp2<
conv_1/Conv2D/ReadVariableOpconv_1/Conv2D/ReadVariableOp2>
conv_2/BiasAdd/ReadVariableOpconv_2/BiasAdd/ReadVariableOp2<
conv_2/Conv2D/ReadVariableOpconv_2/Conv2D/ReadVariableOp2>
conv_3/BiasAdd/ReadVariableOpconv_3/BiasAdd/ReadVariableOp2<
conv_3/Conv2D/ReadVariableOpconv_3/Conv2D/ReadVariableOp2>
conv_4/BiasAdd/ReadVariableOpconv_4/BiasAdd/ReadVariableOp2<
conv_4/Conv2D/ReadVariableOpconv_4/Conv2D/ReadVariableOp2v
9input_batch_normalization/FusedBatchNormV3/ReadVariableOp9input_batch_normalization/FusedBatchNormV3/ReadVariableOp2z
;input_batch_normalization/FusedBatchNormV3/ReadVariableOp_1;input_batch_normalization/FusedBatchNormV3/ReadVariableOp_12T
(input_batch_normalization/ReadVariableOp(input_batch_normalization/ReadVariableOp2X
*input_batch_normalization/ReadVariableOp_1*input_batch_normalization/ReadVariableOp_12F
!input_conv/BiasAdd/ReadVariableOp!input_conv/BiasAdd/ReadVariableOp2D
 input_conv/Conv2D/ReadVariableOp input_conv/Conv2D/ReadVariableOp2d
0output_class_distribution/BiasAdd/ReadVariableOp0output_class_distribution/BiasAdd/ReadVariableOp2b
/output_class_distribution/MatMul/ReadVariableOp/output_class_distribution/MatMul/ReadVariableOp:& "
 
_user_specified_nameinputs
�
D
(__inference_RELU_1_layer_call_fn_4354637

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*/
_output_shapes
:���������@@*-
config_proto

GPU

CPU2*0J 8*L
fGRE
C__inference_RELU_1_layer_call_and_return_conditional_losses_43530772
PartitionedCallt
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:���������@@2

Identity"
identityIdentity:output:0*.
_input_shapes
:���������@@:& "
 
_user_specified_nameinputs
�	
�
V__inference_output_class_distribution_layer_call_and_return_conditional_losses_4355203

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
��*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
MatMul�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2	
BiasAdd�
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*(
_output_shapes
:����������2

Identity"
identityIdentity:output:0*/
_input_shapes
:����������::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:& "
 
_user_specified_nameinputs
�
�
*__inference_b_norm_1_layer_call_fn_4354627

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4*
Tin	
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*A
_output_shapes/
-:+���������������������������*-
config_proto

GPU

CPU2*0J 8*N
fIRG
E__inference_b_norm_1_layer_call_and_return_conditional_losses_43523412
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+���������������������������2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+���������������������������::::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs
�
�
*__inference_b_norm_4_layer_call_fn_4355128

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4*
Tin	
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*/
_output_shapes
:���������@*-
config_proto

GPU

CPU2*0J 8*N
fIRG
E__inference_b_norm_4_layer_call_and_return_conditional_losses_43533142
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:���������@2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:���������@::::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs
�
e
,__inference_dropout_42_layer_call_fn_4354462

inputs
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*/
_output_shapes
:���������@@*-
config_proto

GPU

CPU2*0J 8*P
fKRI
G__inference_dropout_42_layer_call_and_return_conditional_losses_43529722
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:���������@@2

Identity"
identityIdentity:output:0*.
_input_shapes
:���������@@22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs"�L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*�
serving_default�
W
input_conv_inputC
"serving_default_input_conv_input:0�����������N
output_class_distribution1
StatefulPartitionedCall:0����������tensorflow/serving/predict:��
��
layer-0
layer_with_weights-0
layer-1
layer_with_weights-1
layer-2
layer-3
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
layer_with_weights-5
layer-11
layer-12
layer-13
layer_with_weights-6
layer-14
layer_with_weights-7
layer-15
layer-16
layer-17
layer_with_weights-8
layer-18
layer_with_weights-9
layer-19
layer-20
layer-21
layer-22
layer-23
layer_with_weights-10
layer-24
	optimizer
trainable_variables
regularization_losses
	variables
	keras_api

signatures
+�&call_and_return_all_conditional_losses
�_default_save_signature
�__call__"׈
_tf_keras_sequential��{"class_name": "Sequential", "name": "awsome_net", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "config": {"name": "awsome_net", "layers": [{"class_name": "Conv2D", "config": {"name": "input_conv", "trainable": true, "batch_input_shape": [null, 128, 128, 3], "dtype": "float32", "filters": 16, "kernel_size": [3, 3], "strides": [1, 1], "padding": "same", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "BatchNormalization", "config": {"name": "input_batch_normalization", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}}, {"class_name": "ReLU", "config": {"name": "input_RELU", "trainable": true, "dtype": "float32", "max_value": null, "negative_slope": 0.0, "threshold": 0.0}}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_85", "trainable": true, "dtype": "float32", "pool_size": [2, 2], "padding": "valid", "strides": [2, 2], "data_format": "channels_last"}}, {"class_name": "Dropout", "config": {"name": "dropout_42", "trainable": true, "dtype": "float32", "rate": 0.35, "noise_shape": null, "seed": null}}, {"class_name": "Conv2D", "config": {"name": "conv_1", "trainable": true, "dtype": "float32", "filters": 16, "kernel_size": [3, 3], "strides": [1, 1], "padding": "same", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "BatchNormalization", "config": {"name": "b_norm_1", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}}, {"class_name": "ReLU", "config": {"name": "RELU_1", "trainable": true, "dtype": "float32", "max_value": null, "negative_slope": 0.0, "threshold": 0.0}}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_81", "trainable": true, "dtype": "float32", "pool_size": [2, 2], "padding": "valid", "strides": [2, 2], "data_format": "channels_last"}}, {"class_name": "Conv2D", "config": {"name": "conv_2", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": [3, 3], "strides": [1, 1], "padding": "same", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "BatchNormalization", "config": {"name": "b_norm_2", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}}, {"class_name": "ReLU", "config": {"name": "RELU_2", "trainable": true, "dtype": "float32", "max_value": null, "negative_slope": 0.0, "threshold": 0.0}}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_82", "trainable": true, "dtype": "float32", "pool_size": [2, 2], "padding": "valid", "strides": [2, 2], "data_format": "channels_last"}}, {"class_name": "Conv2D", "config": {"name": "conv_3", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": [3, 3], "strides": [1, 1], "padding": "same", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "BatchNormalization", "config": {"name": "b_norm_3", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}}, {"class_name": "ReLU", "config": {"name": "RELU_3", "trainable": true, "dtype": "float32", "max_value": null, "negative_slope": 0.0, "threshold": 0.0}}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_83", "trainable": true, "dtype": "float32", "pool_size": [2, 2], "padding": "valid", "strides": [2, 2], "data_format": "channels_last"}}, {"class_name": "Conv2D", "config": {"name": "conv_4", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": [3, 3], "strides": [1, 1], "padding": "same", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "BatchNormalization", "config": {"name": "b_norm_4", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}}, {"class_name": "ReLU", "config": {"name": "RELU_4", "trainable": true, "dtype": "float32", "max_value": null, "negative_slope": 0.0, "threshold": 0.0}}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_84", "trainable": true, "dtype": "float32", "pool_size": [2, 2], "padding": "valid", "strides": [2, 2], "data_format": "channels_last"}}, {"class_name": "Flatten", "config": {"name": "flatten_21", "trainable": true, "dtype": "float32", "data_format": "channels_last"}}, {"class_name": "Dropout", "config": {"name": "dropout_43", "trainable": true, "dtype": "float32", "rate": 0.35, "noise_shape": null, "seed": null}}, {"class_name": "Dense", "config": {"name": "output_class_distribution", "trainable": true, "dtype": "float32", "units": 200, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}]}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"-1": 3}}}, "is_graph_network": true, "keras_version": "2.2.4-tf", "backend": "tensorflow", "model_config": {"class_name": "Sequential", "config": {"name": "awsome_net", "layers": [{"class_name": "Conv2D", "config": {"name": "input_conv", "trainable": true, "batch_input_shape": [null, 128, 128, 3], "dtype": "float32", "filters": 16, "kernel_size": [3, 3], "strides": [1, 1], "padding": "same", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "BatchNormalization", "config": {"name": "input_batch_normalization", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}}, {"class_name": "ReLU", "config": {"name": "input_RELU", "trainable": true, "dtype": "float32", "max_value": null, "negative_slope": 0.0, "threshold": 0.0}}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_85", "trainable": true, "dtype": "float32", "pool_size": [2, 2], "padding": "valid", "strides": [2, 2], "data_format": "channels_last"}}, {"class_name": "Dropout", "config": {"name": "dropout_42", "trainable": true, "dtype": "float32", "rate": 0.35, "noise_shape": null, "seed": null}}, {"class_name": "Conv2D", "config": {"name": "conv_1", "trainable": true, "dtype": "float32", "filters": 16, "kernel_size": [3, 3], "strides": [1, 1], "padding": "same", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "BatchNormalization", "config": {"name": "b_norm_1", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}}, {"class_name": "ReLU", "config": {"name": "RELU_1", "trainable": true, "dtype": "float32", "max_value": null, "negative_slope": 0.0, "threshold": 0.0}}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_81", "trainable": true, "dtype": "float32", "pool_size": [2, 2], "padding": "valid", "strides": [2, 2], "data_format": "channels_last"}}, {"class_name": "Conv2D", "config": {"name": "conv_2", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": [3, 3], "strides": [1, 1], "padding": "same", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "BatchNormalization", "config": {"name": "b_norm_2", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}}, {"class_name": "ReLU", "config": {"name": "RELU_2", "trainable": true, "dtype": "float32", "max_value": null, "negative_slope": 0.0, "threshold": 0.0}}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_82", "trainable": true, "dtype": "float32", "pool_size": [2, 2], "padding": "valid", "strides": [2, 2], "data_format": "channels_last"}}, {"class_name": "Conv2D", "config": {"name": "conv_3", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": [3, 3], "strides": [1, 1], "padding": "same", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "BatchNormalization", "config": {"name": "b_norm_3", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}}, {"class_name": "ReLU", "config": {"name": "RELU_3", "trainable": true, "dtype": "float32", "max_value": null, "negative_slope": 0.0, "threshold": 0.0}}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_83", "trainable": true, "dtype": "float32", "pool_size": [2, 2], "padding": "valid", "strides": [2, 2], "data_format": "channels_last"}}, {"class_name": "Conv2D", "config": {"name": "conv_4", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": [3, 3], "strides": [1, 1], "padding": "same", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "BatchNormalization", "config": {"name": "b_norm_4", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}}, {"class_name": "ReLU", "config": {"name": "RELU_4", "trainable": true, "dtype": "float32", "max_value": null, "negative_slope": 0.0, "threshold": 0.0}}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_84", "trainable": true, "dtype": "float32", "pool_size": [2, 2], "padding": "valid", "strides": [2, 2], "data_format": "channels_last"}}, {"class_name": "Flatten", "config": {"name": "flatten_21", "trainable": true, "dtype": "float32", "data_format": "channels_last"}}, {"class_name": "Dropout", "config": {"name": "dropout_43", "trainable": true, "dtype": "float32", "rate": 0.35, "noise_shape": null, "seed": null}}, {"class_name": "Dense", "config": {"name": "output_class_distribution", "trainable": true, "dtype": "float32", "units": 200, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}]}}, "training_config": {"loss": {"class_name": "CategoricalCrossentropy", "config": {"reduction": "auto", "name": "categorical_crossentropy", "from_logits": true, "label_smoothing": 0}}, "metrics": [{"class_name": "CategoricalAccuracy", "config": {"name": "categorical_accuracy", "dtype": "float32"}}], "weighted_metrics": null, "sample_weight_mode": null, "loss_weights": null, "optimizer_config": {"class_name": "Adam", "config": {"name": "Adam", "learning_rate": 0.00067, "decay": 0.0, "beta_1": 0.8999999761581421, "beta_2": 0.9990000128746033, "epsilon": 1e-07, "amsgrad": false}}}}
�"�
_tf_keras_input_layer�{"class_name": "InputLayer", "name": "input_conv_input", "dtype": "float32", "sparse": false, "ragged": false, "batch_input_shape": [null, 128, 128, 3], "config": {"batch_input_shape": [null, 128, 128, 3], "dtype": "float32", "sparse": false, "ragged": false, "name": "input_conv_input"}}
�

 kernel
!bias
"trainable_variables
#regularization_losses
$	variables
%	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�
_tf_keras_layer�{"class_name": "Conv2D", "name": "input_conv", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": [null, 128, 128, 3], "config": {"name": "input_conv", "trainable": true, "batch_input_shape": [null, 128, 128, 3], "dtype": "float32", "filters": 16, "kernel_size": [3, 3], "strides": [1, 1], "padding": "same", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"-1": 3}}}}
�
&axis
	'gamma
(beta
)moving_mean
*moving_variance
+trainable_variables
,regularization_losses
-	variables
.	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�
_tf_keras_layer�{"class_name": "BatchNormalization", "name": "input_batch_normalization", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "config": {"name": "input_batch_normalization", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"3": 16}}}}
�
/trainable_variables
0regularization_losses
1	variables
2	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�
_tf_keras_layer�{"class_name": "ReLU", "name": "input_RELU", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "input_RELU", "trainable": true, "dtype": "float32", "max_value": null, "negative_slope": 0.0, "threshold": 0.0}}
�
3trainable_variables
4regularization_losses
5	variables
6	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�
_tf_keras_layer�{"class_name": "MaxPooling2D", "name": "max_pooling2d_85", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "max_pooling2d_85", "trainable": true, "dtype": "float32", "pool_size": [2, 2], "padding": "valid", "strides": [2, 2], "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}}
�
7trainable_variables
8regularization_losses
9	variables
:	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�
_tf_keras_layer�{"class_name": "Dropout", "name": "dropout_42", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "config": {"name": "dropout_42", "trainable": true, "dtype": "float32", "rate": 0.35, "noise_shape": null, "seed": null}}
�

;kernel
<bias
=trainable_variables
>regularization_losses
?	variables
@	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�
_tf_keras_layer�{"class_name": "Conv2D", "name": "conv_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "conv_1", "trainable": true, "dtype": "float32", "filters": 16, "kernel_size": [3, 3], "strides": [1, 1], "padding": "same", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"-1": 16}}}}
�
Aaxis
	Bgamma
Cbeta
Dmoving_mean
Emoving_variance
Ftrainable_variables
Gregularization_losses
H	variables
I	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�
_tf_keras_layer�{"class_name": "BatchNormalization", "name": "b_norm_1", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "config": {"name": "b_norm_1", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"3": 16}}}}
�
Jtrainable_variables
Kregularization_losses
L	variables
M	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�
_tf_keras_layer�{"class_name": "ReLU", "name": "RELU_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "RELU_1", "trainable": true, "dtype": "float32", "max_value": null, "negative_slope": 0.0, "threshold": 0.0}}
�
Ntrainable_variables
Oregularization_losses
P	variables
Q	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�
_tf_keras_layer�{"class_name": "MaxPooling2D", "name": "max_pooling2d_81", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "max_pooling2d_81", "trainable": true, "dtype": "float32", "pool_size": [2, 2], "padding": "valid", "strides": [2, 2], "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}}
�

Rkernel
Sbias
Ttrainable_variables
Uregularization_losses
V	variables
W	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�
_tf_keras_layer�{"class_name": "Conv2D", "name": "conv_2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "conv_2", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": [3, 3], "strides": [1, 1], "padding": "same", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"-1": 16}}}}
�
Xaxis
	Ygamma
Zbeta
[moving_mean
\moving_variance
]trainable_variables
^regularization_losses
_	variables
`	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�
_tf_keras_layer�{"class_name": "BatchNormalization", "name": "b_norm_2", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "config": {"name": "b_norm_2", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"3": 32}}}}
�
atrainable_variables
bregularization_losses
c	variables
d	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�
_tf_keras_layer�{"class_name": "ReLU", "name": "RELU_2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "RELU_2", "trainable": true, "dtype": "float32", "max_value": null, "negative_slope": 0.0, "threshold": 0.0}}
�
etrainable_variables
fregularization_losses
g	variables
h	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�
_tf_keras_layer�{"class_name": "MaxPooling2D", "name": "max_pooling2d_82", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "max_pooling2d_82", "trainable": true, "dtype": "float32", "pool_size": [2, 2], "padding": "valid", "strides": [2, 2], "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}}
�

ikernel
jbias
ktrainable_variables
lregularization_losses
m	variables
n	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�
_tf_keras_layer�{"class_name": "Conv2D", "name": "conv_3", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "conv_3", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": [3, 3], "strides": [1, 1], "padding": "same", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"-1": 32}}}}
�
oaxis
	pgamma
qbeta
rmoving_mean
smoving_variance
ttrainable_variables
uregularization_losses
v	variables
w	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�
_tf_keras_layer�{"class_name": "BatchNormalization", "name": "b_norm_3", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "config": {"name": "b_norm_3", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"3": 64}}}}
�
xtrainable_variables
yregularization_losses
z	variables
{	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�
_tf_keras_layer�{"class_name": "ReLU", "name": "RELU_3", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "RELU_3", "trainable": true, "dtype": "float32", "max_value": null, "negative_slope": 0.0, "threshold": 0.0}}
�
|trainable_variables
}regularization_losses
~	variables
	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�
_tf_keras_layer�{"class_name": "MaxPooling2D", "name": "max_pooling2d_83", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "max_pooling2d_83", "trainable": true, "dtype": "float32", "pool_size": [2, 2], "padding": "valid", "strides": [2, 2], "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}}
�
�kernel
	�bias
�trainable_variables
�regularization_losses
�	variables
�	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�
_tf_keras_layer�{"class_name": "Conv2D", "name": "conv_4", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "conv_4", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": [3, 3], "strides": [1, 1], "padding": "same", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"-1": 64}}}}
�
	�axis

�gamma
	�beta
�moving_mean
�moving_variance
�trainable_variables
�regularization_losses
�	variables
�	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�
_tf_keras_layer�{"class_name": "BatchNormalization", "name": "b_norm_4", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "config": {"name": "b_norm_4", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"3": 64}}}}
�
�trainable_variables
�regularization_losses
�	variables
�	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�
_tf_keras_layer�{"class_name": "ReLU", "name": "RELU_4", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "RELU_4", "trainable": true, "dtype": "float32", "max_value": null, "negative_slope": 0.0, "threshold": 0.0}}
�
�trainable_variables
�regularization_losses
�	variables
�	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�
_tf_keras_layer�{"class_name": "MaxPooling2D", "name": "max_pooling2d_84", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "max_pooling2d_84", "trainable": true, "dtype": "float32", "pool_size": [2, 2], "padding": "valid", "strides": [2, 2], "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}}
�
�trainable_variables
�regularization_losses
�	variables
�	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�
_tf_keras_layer�{"class_name": "Flatten", "name": "flatten_21", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "flatten_21", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 1, "axes": {}}}}
�
�trainable_variables
�regularization_losses
�	variables
�	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�
_tf_keras_layer�{"class_name": "Dropout", "name": "dropout_43", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "config": {"name": "dropout_43", "trainable": true, "dtype": "float32", "rate": 0.35, "noise_shape": null, "seed": null}}
�
�kernel
	�bias
�trainable_variables
�regularization_losses
�	variables
�	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�
_tf_keras_layer�{"class_name": "Dense", "name": "output_class_distribution", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "output_class_distribution", "trainable": true, "dtype": "float32", "units": 200, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 1024}}}}
�
	�iter
�beta_1
�beta_2

�decay
�learning_rate m�!m�'m�(m�;m�<m�Bm�Cm�Rm�Sm�Ym�Zm�im�jm�pm�qm�	�m�	�m�	�m�	�m�	�m�	�m� v�!v�'v�(v�;v�<v�Bv�Cv�Rv�Sv�Yv�Zv�iv�jv�pv�qv�	�v�	�v�	�v�	�v�	�v�	�v�"
	optimizer
�
 0
!1
'2
(3
;4
<5
B6
C7
R8
S9
Y10
Z11
i12
j13
p14
q15
�16
�17
�18
�19
�20
�21"
trackable_list_wrapper
 "
trackable_list_wrapper
�
 0
!1
'2
(3
)4
*5
;6
<7
B8
C9
D10
E11
R12
S13
Y14
Z15
[16
\17
i18
j19
p20
q21
r22
s23
�24
�25
�26
�27
�28
�29
�30
�31"
trackable_list_wrapper
�
 �layer_regularization_losses
trainable_variables
regularization_losses
�non_trainable_variables
�layers
	variables
�metrics
�__call__
�_default_save_signature
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
-
�serving_default"
signature_map
.:,2input_conv_21/kernel
 :2input_conv_21/bias
.
 0
!1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
 0
!1"
trackable_list_wrapper
�
�metrics
"trainable_variables
#regularization_losses
�non_trainable_variables
�layers
$	variables
 �layer_regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
0:.2"input_batch_normalization_21/gamma
/:-2!input_batch_normalization_21/beta
8:6 (2(input_batch_normalization_21/moving_mean
<:: (2,input_batch_normalization_21/moving_variance
.
'0
(1"
trackable_list_wrapper
 "
trackable_list_wrapper
<
'0
(1
)2
*3"
trackable_list_wrapper
�
�metrics
+trainable_variables
,regularization_losses
�non_trainable_variables
�layers
-	variables
 �layer_regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�metrics
/trainable_variables
0regularization_losses
�non_trainable_variables
�layers
1	variables
 �layer_regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�metrics
3trainable_variables
4regularization_losses
�non_trainable_variables
�layers
5	variables
 �layer_regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�metrics
7trainable_variables
8regularization_losses
�non_trainable_variables
�layers
9	variables
 �layer_regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
*:(2conv_1_21/kernel
:2conv_1_21/bias
.
;0
<1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
;0
<1"
trackable_list_wrapper
�
�metrics
=trainable_variables
>regularization_losses
�non_trainable_variables
�layers
?	variables
 �layer_regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
:2b_norm_1_21/gamma
:2b_norm_1_21/beta
':% (2b_norm_1_21/moving_mean
+:) (2b_norm_1_21/moving_variance
.
B0
C1"
trackable_list_wrapper
 "
trackable_list_wrapper
<
B0
C1
D2
E3"
trackable_list_wrapper
�
�metrics
Ftrainable_variables
Gregularization_losses
�non_trainable_variables
�layers
H	variables
 �layer_regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�metrics
Jtrainable_variables
Kregularization_losses
�non_trainable_variables
�layers
L	variables
 �layer_regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�metrics
Ntrainable_variables
Oregularization_losses
�non_trainable_variables
�layers
P	variables
 �layer_regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
*:( 2conv_2_21/kernel
: 2conv_2_21/bias
.
R0
S1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
R0
S1"
trackable_list_wrapper
�
�metrics
Ttrainable_variables
Uregularization_losses
�non_trainable_variables
�layers
V	variables
 �layer_regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
: 2b_norm_2_21/gamma
: 2b_norm_2_21/beta
':%  (2b_norm_2_21/moving_mean
+:)  (2b_norm_2_21/moving_variance
.
Y0
Z1"
trackable_list_wrapper
 "
trackable_list_wrapper
<
Y0
Z1
[2
\3"
trackable_list_wrapper
�
�metrics
]trainable_variables
^regularization_losses
�non_trainable_variables
�layers
_	variables
 �layer_regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�metrics
atrainable_variables
bregularization_losses
�non_trainable_variables
�layers
c	variables
 �layer_regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�metrics
etrainable_variables
fregularization_losses
�non_trainable_variables
�layers
g	variables
 �layer_regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
*:( @2conv_3_13/kernel
:@2conv_3_13/bias
.
i0
j1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
i0
j1"
trackable_list_wrapper
�
�metrics
ktrainable_variables
lregularization_losses
�non_trainable_variables
�layers
m	variables
 �layer_regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
:@2b_norm_3_13/gamma
:@2b_norm_3_13/beta
':%@ (2b_norm_3_13/moving_mean
+:)@ (2b_norm_3_13/moving_variance
.
p0
q1"
trackable_list_wrapper
 "
trackable_list_wrapper
<
p0
q1
r2
s3"
trackable_list_wrapper
�
�metrics
ttrainable_variables
uregularization_losses
�non_trainable_variables
�layers
v	variables
 �layer_regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�metrics
xtrainable_variables
yregularization_losses
�non_trainable_variables
�layers
z	variables
 �layer_regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�metrics
|trainable_variables
}regularization_losses
�non_trainable_variables
�layers
~	variables
 �layer_regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
):'@@2conv_4_5/kernel
:@2conv_4_5/bias
0
�0
�1"
trackable_list_wrapper
 "
trackable_list_wrapper
0
�0
�1"
trackable_list_wrapper
�
�metrics
�trainable_variables
�regularization_losses
�non_trainable_variables
�layers
�	variables
 �layer_regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
:@2b_norm_4_5/gamma
:@2b_norm_4_5/beta
&:$@ (2b_norm_4_5/moving_mean
*:(@ (2b_norm_4_5/moving_variance
0
�0
�1"
trackable_list_wrapper
 "
trackable_list_wrapper
@
�0
�1
�2
�3"
trackable_list_wrapper
�
�metrics
�trainable_variables
�regularization_losses
�non_trainable_variables
�layers
�	variables
 �layer_regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�metrics
�trainable_variables
�regularization_losses
�non_trainable_variables
�layers
�	variables
 �layer_regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�metrics
�trainable_variables
�regularization_losses
�non_trainable_variables
�layers
�	variables
 �layer_regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�metrics
�trainable_variables
�regularization_losses
�non_trainable_variables
�layers
�	variables
 �layer_regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�metrics
�trainable_variables
�regularization_losses
�non_trainable_variables
�layers
�	variables
 �layer_regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
7:5
��2#output_class_distribution_21/kernel
0:.�2!output_class_distribution_21/bias
0
�0
�1"
trackable_list_wrapper
 "
trackable_list_wrapper
0
�0
�1"
trackable_list_wrapper
�
�metrics
�trainable_variables
�regularization_losses
�non_trainable_variables
�layers
�	variables
 �layer_regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
:	 (2	Adam/iter
: (2Adam/beta_1
: (2Adam/beta_2
: (2
Adam/decay
: (2Adam/learning_rate
 "
trackable_list_wrapper
h
)0
*1
D2
E3
[4
\5
r6
s7
�8
�9"
trackable_list_wrapper
�
0
1
2
3
4
5
6
	7

8
9
10
11
12
13
14
15
16
17
18
19
20
21
22
23"
trackable_list_wrapper
(
�0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
.
)0
*1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
.
D0
E1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
.
[0
\1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
.
r0
s1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
0
�0
�1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�

�total

�count
�
_fn_kwargs
�trainable_variables
�regularization_losses
�	variables
�	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�
_tf_keras_layer�{"class_name": "CategoricalAccuracy", "name": "categorical_accuracy", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "config": {"name": "categorical_accuracy", "dtype": "float32"}}
:  (2total
:  (2count
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
0
�0
�1"
trackable_list_wrapper
�
�metrics
�trainable_variables
�regularization_losses
�non_trainable_variables
�layers
�	variables
 �layer_regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
0
�0
�1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
3:12Adam/input_conv_21/kernel/m
%:#2Adam/input_conv_21/bias/m
5:32)Adam/input_batch_normalization_21/gamma/m
4:22(Adam/input_batch_normalization_21/beta/m
/:-2Adam/conv_1_21/kernel/m
!:2Adam/conv_1_21/bias/m
$:"2Adam/b_norm_1_21/gamma/m
#:!2Adam/b_norm_1_21/beta/m
/:- 2Adam/conv_2_21/kernel/m
!: 2Adam/conv_2_21/bias/m
$:" 2Adam/b_norm_2_21/gamma/m
#:! 2Adam/b_norm_2_21/beta/m
/:- @2Adam/conv_3_13/kernel/m
!:@2Adam/conv_3_13/bias/m
$:"@2Adam/b_norm_3_13/gamma/m
#:!@2Adam/b_norm_3_13/beta/m
.:,@@2Adam/conv_4_5/kernel/m
 :@2Adam/conv_4_5/bias/m
#:!@2Adam/b_norm_4_5/gamma/m
": @2Adam/b_norm_4_5/beta/m
<::
��2*Adam/output_class_distribution_21/kernel/m
5:3�2(Adam/output_class_distribution_21/bias/m
3:12Adam/input_conv_21/kernel/v
%:#2Adam/input_conv_21/bias/v
5:32)Adam/input_batch_normalization_21/gamma/v
4:22(Adam/input_batch_normalization_21/beta/v
/:-2Adam/conv_1_21/kernel/v
!:2Adam/conv_1_21/bias/v
$:"2Adam/b_norm_1_21/gamma/v
#:!2Adam/b_norm_1_21/beta/v
/:- 2Adam/conv_2_21/kernel/v
!: 2Adam/conv_2_21/bias/v
$:" 2Adam/b_norm_2_21/gamma/v
#:! 2Adam/b_norm_2_21/beta/v
/:- @2Adam/conv_3_13/kernel/v
!:@2Adam/conv_3_13/bias/v
$:"@2Adam/b_norm_3_13/gamma/v
#:!@2Adam/b_norm_3_13/beta/v
.:,@@2Adam/conv_4_5/kernel/v
 :@2Adam/conv_4_5/bias/v
#:!@2Adam/b_norm_4_5/gamma/v
": @2Adam/b_norm_4_5/beta/v
<::
��2*Adam/output_class_distribution_21/kernel/v
5:3�2(Adam/output_class_distribution_21/bias/v
�2�
G__inference_awsome_net_layer_call_and_return_conditional_losses_4353449
G__inference_awsome_net_layer_call_and_return_conditional_losses_4353509
G__inference_awsome_net_layer_call_and_return_conditional_losses_4354188
G__inference_awsome_net_layer_call_and_return_conditional_losses_4354044�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
"__inference__wrapped_model_4352032�
���
FullArgSpec
args� 
varargsjargs
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *9�6
4�1
input_conv_input�����������
�2�
,__inference_awsome_net_layer_call_fn_4353704
,__inference_awsome_net_layer_call_fn_4354262
,__inference_awsome_net_layer_call_fn_4354225
,__inference_awsome_net_layer_call_fn_4353607�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
G__inference_input_conv_layer_call_and_return_conditional_losses_4352044�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *7�4
2�/+���������������������������
�2�
,__inference_input_conv_layer_call_fn_4352052�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *7�4
2�/+���������������������������
�2�
V__inference_input_batch_normalization_layer_call_and_return_conditional_losses_4354330
V__inference_input_batch_normalization_layer_call_and_return_conditional_losses_4354382
V__inference_input_batch_normalization_layer_call_and_return_conditional_losses_4354404
V__inference_input_batch_normalization_layer_call_and_return_conditional_losses_4354308�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
;__inference_input_batch_normalization_layer_call_fn_4354413
;__inference_input_batch_normalization_layer_call_fn_4354348
;__inference_input_batch_normalization_layer_call_fn_4354422
;__inference_input_batch_normalization_layer_call_fn_4354339�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
G__inference_input_RELU_layer_call_and_return_conditional_losses_4354427�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
,__inference_input_RELU_layer_call_fn_4354432�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
M__inference_max_pooling2d_85_layer_call_and_return_conditional_losses_4352190�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *@�=
;�84������������������������������������
�2�
2__inference_max_pooling2d_85_layer_call_fn_4352196�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *@�=
;�84������������������������������������
�2�
G__inference_dropout_42_layer_call_and_return_conditional_losses_4354452
G__inference_dropout_42_layer_call_and_return_conditional_losses_4354457�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
,__inference_dropout_42_layer_call_fn_4354467
,__inference_dropout_42_layer_call_fn_4354462�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
C__inference_conv_1_layer_call_and_return_conditional_losses_4352208�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *7�4
2�/+���������������������������
�2�
(__inference_conv_1_layer_call_fn_4352216�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *7�4
2�/+���������������������������
�2�
E__inference_b_norm_1_layer_call_and_return_conditional_losses_4354535
E__inference_b_norm_1_layer_call_and_return_conditional_losses_4354609
E__inference_b_norm_1_layer_call_and_return_conditional_losses_4354587
E__inference_b_norm_1_layer_call_and_return_conditional_losses_4354513�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
*__inference_b_norm_1_layer_call_fn_4354553
*__inference_b_norm_1_layer_call_fn_4354627
*__inference_b_norm_1_layer_call_fn_4354544
*__inference_b_norm_1_layer_call_fn_4354618�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
C__inference_RELU_1_layer_call_and_return_conditional_losses_4354632�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
(__inference_RELU_1_layer_call_fn_4354637�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
M__inference_max_pooling2d_81_layer_call_and_return_conditional_losses_4352354�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *@�=
;�84������������������������������������
�2�
2__inference_max_pooling2d_81_layer_call_fn_4352360�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *@�=
;�84������������������������������������
�2�
C__inference_conv_2_layer_call_and_return_conditional_losses_4352372�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *7�4
2�/+���������������������������
�2�
(__inference_conv_2_layer_call_fn_4352380�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *7�4
2�/+���������������������������
�2�
E__inference_b_norm_2_layer_call_and_return_conditional_losses_4354757
E__inference_b_norm_2_layer_call_and_return_conditional_losses_4354705
E__inference_b_norm_2_layer_call_and_return_conditional_losses_4354779
E__inference_b_norm_2_layer_call_and_return_conditional_losses_4354683�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
*__inference_b_norm_2_layer_call_fn_4354788
*__inference_b_norm_2_layer_call_fn_4354797
*__inference_b_norm_2_layer_call_fn_4354723
*__inference_b_norm_2_layer_call_fn_4354714�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
C__inference_RELU_2_layer_call_and_return_conditional_losses_4354802�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
(__inference_RELU_2_layer_call_fn_4354807�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
M__inference_max_pooling2d_82_layer_call_and_return_conditional_losses_4352518�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *@�=
;�84������������������������������������
�2�
2__inference_max_pooling2d_82_layer_call_fn_4352524�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *@�=
;�84������������������������������������
�2�
C__inference_conv_3_layer_call_and_return_conditional_losses_4352536�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *7�4
2�/+��������������������������� 
�2�
(__inference_conv_3_layer_call_fn_4352544�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *7�4
2�/+��������������������������� 
�2�
E__inference_b_norm_3_layer_call_and_return_conditional_losses_4354875
E__inference_b_norm_3_layer_call_and_return_conditional_losses_4354927
E__inference_b_norm_3_layer_call_and_return_conditional_losses_4354853
E__inference_b_norm_3_layer_call_and_return_conditional_losses_4354949�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
*__inference_b_norm_3_layer_call_fn_4354893
*__inference_b_norm_3_layer_call_fn_4354884
*__inference_b_norm_3_layer_call_fn_4354958
*__inference_b_norm_3_layer_call_fn_4354967�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
C__inference_RELU_3_layer_call_and_return_conditional_losses_4354972�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
(__inference_RELU_3_layer_call_fn_4354977�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
M__inference_max_pooling2d_83_layer_call_and_return_conditional_losses_4352682�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *@�=
;�84������������������������������������
�2�
2__inference_max_pooling2d_83_layer_call_fn_4352688�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *@�=
;�84������������������������������������
�2�
C__inference_conv_4_layer_call_and_return_conditional_losses_4352700�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *7�4
2�/+���������������������������@
�2�
(__inference_conv_4_layer_call_fn_4352708�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *7�4
2�/+���������������������������@
�2�
E__inference_b_norm_4_layer_call_and_return_conditional_losses_4355023
E__inference_b_norm_4_layer_call_and_return_conditional_losses_4355045
E__inference_b_norm_4_layer_call_and_return_conditional_losses_4355097
E__inference_b_norm_4_layer_call_and_return_conditional_losses_4355119�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
*__inference_b_norm_4_layer_call_fn_4355128
*__inference_b_norm_4_layer_call_fn_4355054
*__inference_b_norm_4_layer_call_fn_4355063
*__inference_b_norm_4_layer_call_fn_4355137�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
C__inference_RELU_4_layer_call_and_return_conditional_losses_4355142�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
(__inference_RELU_4_layer_call_fn_4355147�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
M__inference_max_pooling2d_84_layer_call_and_return_conditional_losses_4352846�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *@�=
;�84������������������������������������
�2�
2__inference_max_pooling2d_84_layer_call_fn_4352852�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *@�=
;�84������������������������������������
�2�
G__inference_flatten_21_layer_call_and_return_conditional_losses_4355153�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
,__inference_flatten_21_layer_call_fn_4355158�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
G__inference_dropout_43_layer_call_and_return_conditional_losses_4355183
G__inference_dropout_43_layer_call_and_return_conditional_losses_4355178�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
,__inference_dropout_43_layer_call_fn_4355188
,__inference_dropout_43_layer_call_fn_4355193�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
V__inference_output_class_distribution_layer_call_and_return_conditional_losses_4355203�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
;__inference_output_class_distribution_layer_call_fn_4355210�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
=B;
%__inference_signature_wrapper_4353810input_conv_input
�2��
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkwjkwargs
defaults� 

kwonlyargs�

jtraining%
kwonlydefaults�

trainingp 
annotations� *
 
�2��
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkwjkwargs
defaults� 

kwonlyargs�

jtraining%
kwonlydefaults�

trainingp 
annotations� *
 �
C__inference_RELU_1_layer_call_and_return_conditional_losses_4354632h7�4
-�*
(�%
inputs���������@@
� "-�*
#� 
0���������@@
� �
(__inference_RELU_1_layer_call_fn_4354637[7�4
-�*
(�%
inputs���������@@
� " ����������@@�
C__inference_RELU_2_layer_call_and_return_conditional_losses_4354802h7�4
-�*
(�%
inputs���������   
� "-�*
#� 
0���������   
� �
(__inference_RELU_2_layer_call_fn_4354807[7�4
-�*
(�%
inputs���������   
� " ����������   �
C__inference_RELU_3_layer_call_and_return_conditional_losses_4354972h7�4
-�*
(�%
inputs���������@
� "-�*
#� 
0���������@
� �
(__inference_RELU_3_layer_call_fn_4354977[7�4
-�*
(�%
inputs���������@
� " ����������@�
C__inference_RELU_4_layer_call_and_return_conditional_losses_4355142h7�4
-�*
(�%
inputs���������@
� "-�*
#� 
0���������@
� �
(__inference_RELU_4_layer_call_fn_4355147[7�4
-�*
(�%
inputs���������@
� " ����������@�
"__inference__wrapped_model_4352032�( !'()*;<BCDERSYZ[\ijpqrs��������C�@
9�6
4�1
input_conv_input�����������
� "V�S
Q
output_class_distribution4�1
output_class_distribution�����������
G__inference_awsome_net_layer_call_and_return_conditional_losses_4353449�( !'()*;<BCDERSYZ[\ijpqrs��������K�H
A�>
4�1
input_conv_input�����������
p

 
� "&�#
�
0����������
� �
G__inference_awsome_net_layer_call_and_return_conditional_losses_4353509�( !'()*;<BCDERSYZ[\ijpqrs��������K�H
A�>
4�1
input_conv_input�����������
p 

 
� "&�#
�
0����������
� �
G__inference_awsome_net_layer_call_and_return_conditional_losses_4354044�( !'()*;<BCDERSYZ[\ijpqrs��������A�>
7�4
*�'
inputs�����������
p

 
� "&�#
�
0����������
� �
G__inference_awsome_net_layer_call_and_return_conditional_losses_4354188�( !'()*;<BCDERSYZ[\ijpqrs��������A�>
7�4
*�'
inputs�����������
p 

 
� "&�#
�
0����������
� �
,__inference_awsome_net_layer_call_fn_4353607�( !'()*;<BCDERSYZ[\ijpqrs��������K�H
A�>
4�1
input_conv_input�����������
p

 
� "������������
,__inference_awsome_net_layer_call_fn_4353704�( !'()*;<BCDERSYZ[\ijpqrs��������K�H
A�>
4�1
input_conv_input�����������
p 

 
� "������������
,__inference_awsome_net_layer_call_fn_4354225�( !'()*;<BCDERSYZ[\ijpqrs��������A�>
7�4
*�'
inputs�����������
p

 
� "������������
,__inference_awsome_net_layer_call_fn_4354262�( !'()*;<BCDERSYZ[\ijpqrs��������A�>
7�4
*�'
inputs�����������
p 

 
� "������������
E__inference_b_norm_1_layer_call_and_return_conditional_losses_4354513rBCDE;�8
1�.
(�%
inputs���������@@
p
� "-�*
#� 
0���������@@
� �
E__inference_b_norm_1_layer_call_and_return_conditional_losses_4354535rBCDE;�8
1�.
(�%
inputs���������@@
p 
� "-�*
#� 
0���������@@
� �
E__inference_b_norm_1_layer_call_and_return_conditional_losses_4354587�BCDEM�J
C�@
:�7
inputs+���������������������������
p
� "?�<
5�2
0+���������������������������
� �
E__inference_b_norm_1_layer_call_and_return_conditional_losses_4354609�BCDEM�J
C�@
:�7
inputs+���������������������������
p 
� "?�<
5�2
0+���������������������������
� �
*__inference_b_norm_1_layer_call_fn_4354544eBCDE;�8
1�.
(�%
inputs���������@@
p
� " ����������@@�
*__inference_b_norm_1_layer_call_fn_4354553eBCDE;�8
1�.
(�%
inputs���������@@
p 
� " ����������@@�
*__inference_b_norm_1_layer_call_fn_4354618�BCDEM�J
C�@
:�7
inputs+���������������������������
p
� "2�/+����������������������������
*__inference_b_norm_1_layer_call_fn_4354627�BCDEM�J
C�@
:�7
inputs+���������������������������
p 
� "2�/+����������������������������
E__inference_b_norm_2_layer_call_and_return_conditional_losses_4354683�YZ[\M�J
C�@
:�7
inputs+��������������������������� 
p
� "?�<
5�2
0+��������������������������� 
� �
E__inference_b_norm_2_layer_call_and_return_conditional_losses_4354705�YZ[\M�J
C�@
:�7
inputs+��������������������������� 
p 
� "?�<
5�2
0+��������������������������� 
� �
E__inference_b_norm_2_layer_call_and_return_conditional_losses_4354757rYZ[\;�8
1�.
(�%
inputs���������   
p
� "-�*
#� 
0���������   
� �
E__inference_b_norm_2_layer_call_and_return_conditional_losses_4354779rYZ[\;�8
1�.
(�%
inputs���������   
p 
� "-�*
#� 
0���������   
� �
*__inference_b_norm_2_layer_call_fn_4354714�YZ[\M�J
C�@
:�7
inputs+��������������������������� 
p
� "2�/+��������������������������� �
*__inference_b_norm_2_layer_call_fn_4354723�YZ[\M�J
C�@
:�7
inputs+��������������������������� 
p 
� "2�/+��������������������������� �
*__inference_b_norm_2_layer_call_fn_4354788eYZ[\;�8
1�.
(�%
inputs���������   
p
� " ����������   �
*__inference_b_norm_2_layer_call_fn_4354797eYZ[\;�8
1�.
(�%
inputs���������   
p 
� " ����������   �
E__inference_b_norm_3_layer_call_and_return_conditional_losses_4354853�pqrsM�J
C�@
:�7
inputs+���������������������������@
p
� "?�<
5�2
0+���������������������������@
� �
E__inference_b_norm_3_layer_call_and_return_conditional_losses_4354875�pqrsM�J
C�@
:�7
inputs+���������������������������@
p 
� "?�<
5�2
0+���������������������������@
� �
E__inference_b_norm_3_layer_call_and_return_conditional_losses_4354927rpqrs;�8
1�.
(�%
inputs���������@
p
� "-�*
#� 
0���������@
� �
E__inference_b_norm_3_layer_call_and_return_conditional_losses_4354949rpqrs;�8
1�.
(�%
inputs���������@
p 
� "-�*
#� 
0���������@
� �
*__inference_b_norm_3_layer_call_fn_4354884�pqrsM�J
C�@
:�7
inputs+���������������������������@
p
� "2�/+���������������������������@�
*__inference_b_norm_3_layer_call_fn_4354893�pqrsM�J
C�@
:�7
inputs+���������������������������@
p 
� "2�/+���������������������������@�
*__inference_b_norm_3_layer_call_fn_4354958epqrs;�8
1�.
(�%
inputs���������@
p
� " ����������@�
*__inference_b_norm_3_layer_call_fn_4354967epqrs;�8
1�.
(�%
inputs���������@
p 
� " ����������@�
E__inference_b_norm_4_layer_call_and_return_conditional_losses_4355023�����M�J
C�@
:�7
inputs+���������������������������@
p
� "?�<
5�2
0+���������������������������@
� �
E__inference_b_norm_4_layer_call_and_return_conditional_losses_4355045�����M�J
C�@
:�7
inputs+���������������������������@
p 
� "?�<
5�2
0+���������������������������@
� �
E__inference_b_norm_4_layer_call_and_return_conditional_losses_4355097v����;�8
1�.
(�%
inputs���������@
p
� "-�*
#� 
0���������@
� �
E__inference_b_norm_4_layer_call_and_return_conditional_losses_4355119v����;�8
1�.
(�%
inputs���������@
p 
� "-�*
#� 
0���������@
� �
*__inference_b_norm_4_layer_call_fn_4355054�����M�J
C�@
:�7
inputs+���������������������������@
p
� "2�/+���������������������������@�
*__inference_b_norm_4_layer_call_fn_4355063�����M�J
C�@
:�7
inputs+���������������������������@
p 
� "2�/+���������������������������@�
*__inference_b_norm_4_layer_call_fn_4355128i����;�8
1�.
(�%
inputs���������@
p
� " ����������@�
*__inference_b_norm_4_layer_call_fn_4355137i����;�8
1�.
(�%
inputs���������@
p 
� " ����������@�
C__inference_conv_1_layer_call_and_return_conditional_losses_4352208�;<I�F
?�<
:�7
inputs+���������������������������
� "?�<
5�2
0+���������������������������
� �
(__inference_conv_1_layer_call_fn_4352216�;<I�F
?�<
:�7
inputs+���������������������������
� "2�/+����������������������������
C__inference_conv_2_layer_call_and_return_conditional_losses_4352372�RSI�F
?�<
:�7
inputs+���������������������������
� "?�<
5�2
0+��������������������������� 
� �
(__inference_conv_2_layer_call_fn_4352380�RSI�F
?�<
:�7
inputs+���������������������������
� "2�/+��������������������������� �
C__inference_conv_3_layer_call_and_return_conditional_losses_4352536�ijI�F
?�<
:�7
inputs+��������������������������� 
� "?�<
5�2
0+���������������������������@
� �
(__inference_conv_3_layer_call_fn_4352544�ijI�F
?�<
:�7
inputs+��������������������������� 
� "2�/+���������������������������@�
C__inference_conv_4_layer_call_and_return_conditional_losses_4352700���I�F
?�<
:�7
inputs+���������������������������@
� "?�<
5�2
0+���������������������������@
� �
(__inference_conv_4_layer_call_fn_4352708���I�F
?�<
:�7
inputs+���������������������������@
� "2�/+���������������������������@�
G__inference_dropout_42_layer_call_and_return_conditional_losses_4354452l;�8
1�.
(�%
inputs���������@@
p
� "-�*
#� 
0���������@@
� �
G__inference_dropout_42_layer_call_and_return_conditional_losses_4354457l;�8
1�.
(�%
inputs���������@@
p 
� "-�*
#� 
0���������@@
� �
,__inference_dropout_42_layer_call_fn_4354462_;�8
1�.
(�%
inputs���������@@
p
� " ����������@@�
,__inference_dropout_42_layer_call_fn_4354467_;�8
1�.
(�%
inputs���������@@
p 
� " ����������@@�
G__inference_dropout_43_layer_call_and_return_conditional_losses_4355178^4�1
*�'
!�
inputs����������
p
� "&�#
�
0����������
� �
G__inference_dropout_43_layer_call_and_return_conditional_losses_4355183^4�1
*�'
!�
inputs����������
p 
� "&�#
�
0����������
� �
,__inference_dropout_43_layer_call_fn_4355188Q4�1
*�'
!�
inputs����������
p
� "������������
,__inference_dropout_43_layer_call_fn_4355193Q4�1
*�'
!�
inputs����������
p 
� "������������
G__inference_flatten_21_layer_call_and_return_conditional_losses_4355153a7�4
-�*
(�%
inputs���������@
� "&�#
�
0����������
� �
,__inference_flatten_21_layer_call_fn_4355158T7�4
-�*
(�%
inputs���������@
� "������������
G__inference_input_RELU_layer_call_and_return_conditional_losses_4354427l9�6
/�,
*�'
inputs�����������
� "/�,
%�"
0�����������
� �
,__inference_input_RELU_layer_call_fn_4354432_9�6
/�,
*�'
inputs�����������
� ""�������������
V__inference_input_batch_normalization_layer_call_and_return_conditional_losses_4354308�'()*M�J
C�@
:�7
inputs+���������������������������
p
� "?�<
5�2
0+���������������������������
� �
V__inference_input_batch_normalization_layer_call_and_return_conditional_losses_4354330�'()*M�J
C�@
:�7
inputs+���������������������������
p 
� "?�<
5�2
0+���������������������������
� �
V__inference_input_batch_normalization_layer_call_and_return_conditional_losses_4354382v'()*=�:
3�0
*�'
inputs�����������
p
� "/�,
%�"
0�����������
� �
V__inference_input_batch_normalization_layer_call_and_return_conditional_losses_4354404v'()*=�:
3�0
*�'
inputs�����������
p 
� "/�,
%�"
0�����������
� �
;__inference_input_batch_normalization_layer_call_fn_4354339�'()*M�J
C�@
:�7
inputs+���������������������������
p
� "2�/+����������������������������
;__inference_input_batch_normalization_layer_call_fn_4354348�'()*M�J
C�@
:�7
inputs+���������������������������
p 
� "2�/+����������������������������
;__inference_input_batch_normalization_layer_call_fn_4354413i'()*=�:
3�0
*�'
inputs�����������
p
� ""�������������
;__inference_input_batch_normalization_layer_call_fn_4354422i'()*=�:
3�0
*�'
inputs�����������
p 
� ""�������������
G__inference_input_conv_layer_call_and_return_conditional_losses_4352044� !I�F
?�<
:�7
inputs+���������������������������
� "?�<
5�2
0+���������������������������
� �
,__inference_input_conv_layer_call_fn_4352052� !I�F
?�<
:�7
inputs+���������������������������
� "2�/+����������������������������
M__inference_max_pooling2d_81_layer_call_and_return_conditional_losses_4352354�R�O
H�E
C�@
inputs4������������������������������������
� "H�E
>�;
04������������������������������������
� �
2__inference_max_pooling2d_81_layer_call_fn_4352360�R�O
H�E
C�@
inputs4������������������������������������
� ";�84�������������������������������������
M__inference_max_pooling2d_82_layer_call_and_return_conditional_losses_4352518�R�O
H�E
C�@
inputs4������������������������������������
� "H�E
>�;
04������������������������������������
� �
2__inference_max_pooling2d_82_layer_call_fn_4352524�R�O
H�E
C�@
inputs4������������������������������������
� ";�84�������������������������������������
M__inference_max_pooling2d_83_layer_call_and_return_conditional_losses_4352682�R�O
H�E
C�@
inputs4������������������������������������
� "H�E
>�;
04������������������������������������
� �
2__inference_max_pooling2d_83_layer_call_fn_4352688�R�O
H�E
C�@
inputs4������������������������������������
� ";�84�������������������������������������
M__inference_max_pooling2d_84_layer_call_and_return_conditional_losses_4352846�R�O
H�E
C�@
inputs4������������������������������������
� "H�E
>�;
04������������������������������������
� �
2__inference_max_pooling2d_84_layer_call_fn_4352852�R�O
H�E
C�@
inputs4������������������������������������
� ";�84�������������������������������������
M__inference_max_pooling2d_85_layer_call_and_return_conditional_losses_4352190�R�O
H�E
C�@
inputs4������������������������������������
� "H�E
>�;
04������������������������������������
� �
2__inference_max_pooling2d_85_layer_call_fn_4352196�R�O
H�E
C�@
inputs4������������������������������������
� ";�84�������������������������������������
V__inference_output_class_distribution_layer_call_and_return_conditional_losses_4355203`��0�-
&�#
!�
inputs����������
� "&�#
�
0����������
� �
;__inference_output_class_distribution_layer_call_fn_4355210S��0�-
&�#
!�
inputs����������
� "������������
%__inference_signature_wrapper_4353810�( !'()*;<BCDERSYZ[\ijpqrs��������W�T
� 
M�J
H
input_conv_input4�1
input_conv_input�����������"V�S
Q
output_class_distribution4�1
output_class_distribution����������