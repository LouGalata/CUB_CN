НИ#
Щ¤
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
dtypetypeИ
╛
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
executor_typestring И
q
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshapeИ"serve*2.1.02unknown8К╛
М
input_conv_14/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: *%
shared_nameinput_conv_14/kernel
Е
(input_conv_14/kernel/Read/ReadVariableOpReadVariableOpinput_conv_14/kernel*&
_output_shapes
: *
dtype0
|
input_conv_14/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *#
shared_nameinput_conv_14/bias
u
&input_conv_14/bias/Read/ReadVariableOpReadVariableOpinput_conv_14/bias*
_output_shapes
: *
dtype0
Ь
"input_batch_normalization_14/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape: *3
shared_name$"input_batch_normalization_14/gamma
Х
6input_batch_normalization_14/gamma/Read/ReadVariableOpReadVariableOp"input_batch_normalization_14/gamma*
_output_shapes
: *
dtype0
Ъ
!input_batch_normalization_14/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape: *2
shared_name#!input_batch_normalization_14/beta
У
5input_batch_normalization_14/beta/Read/ReadVariableOpReadVariableOp!input_batch_normalization_14/beta*
_output_shapes
: *
dtype0
и
(input_batch_normalization_14/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape: *9
shared_name*(input_batch_normalization_14/moving_mean
б
<input_batch_normalization_14/moving_mean/Read/ReadVariableOpReadVariableOp(input_batch_normalization_14/moving_mean*
_output_shapes
: *
dtype0
░
,input_batch_normalization_14/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape: *=
shared_name.,input_batch_normalization_14/moving_variance
й
@input_batch_normalization_14/moving_variance/Read/ReadVariableOpReadVariableOp,input_batch_normalization_14/moving_variance*
_output_shapes
: *
dtype0
Д
conv_1_14/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:  *!
shared_nameconv_1_14/kernel
}
$conv_1_14/kernel/Read/ReadVariableOpReadVariableOpconv_1_14/kernel*&
_output_shapes
:  *
dtype0
t
conv_1_14/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameconv_1_14/bias
m
"conv_1_14/bias/Read/ReadVariableOpReadVariableOpconv_1_14/bias*
_output_shapes
: *
dtype0
z
b_norm_1_14/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape: *"
shared_nameb_norm_1_14/gamma
s
%b_norm_1_14/gamma/Read/ReadVariableOpReadVariableOpb_norm_1_14/gamma*
_output_shapes
: *
dtype0
x
b_norm_1_14/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape: *!
shared_nameb_norm_1_14/beta
q
$b_norm_1_14/beta/Read/ReadVariableOpReadVariableOpb_norm_1_14/beta*
_output_shapes
: *
dtype0
Ж
b_norm_1_14/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape: *(
shared_nameb_norm_1_14/moving_mean

+b_norm_1_14/moving_mean/Read/ReadVariableOpReadVariableOpb_norm_1_14/moving_mean*
_output_shapes
: *
dtype0
О
b_norm_1_14/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape: *,
shared_nameb_norm_1_14/moving_variance
З
/b_norm_1_14/moving_variance/Read/ReadVariableOpReadVariableOpb_norm_1_14/moving_variance*
_output_shapes
: *
dtype0
Д
conv_2_14/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: @*!
shared_nameconv_2_14/kernel
}
$conv_2_14/kernel/Read/ReadVariableOpReadVariableOpconv_2_14/kernel*&
_output_shapes
: @*
dtype0
t
conv_2_14/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_nameconv_2_14/bias
m
"conv_2_14/bias/Read/ReadVariableOpReadVariableOpconv_2_14/bias*
_output_shapes
:@*
dtype0
z
b_norm_2_14/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*"
shared_nameb_norm_2_14/gamma
s
%b_norm_2_14/gamma/Read/ReadVariableOpReadVariableOpb_norm_2_14/gamma*
_output_shapes
:@*
dtype0
x
b_norm_2_14/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*!
shared_nameb_norm_2_14/beta
q
$b_norm_2_14/beta/Read/ReadVariableOpReadVariableOpb_norm_2_14/beta*
_output_shapes
:@*
dtype0
Ж
b_norm_2_14/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*(
shared_nameb_norm_2_14/moving_mean

+b_norm_2_14/moving_mean/Read/ReadVariableOpReadVariableOpb_norm_2_14/moving_mean*
_output_shapes
:@*
dtype0
О
b_norm_2_14/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*,
shared_nameb_norm_2_14/moving_variance
З
/b_norm_2_14/moving_variance/Read/ReadVariableOpReadVariableOpb_norm_2_14/moving_variance*
_output_shapes
:@*
dtype0
Г
conv_3_9/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@А* 
shared_nameconv_3_9/kernel
|
#conv_3_9/kernel/Read/ReadVariableOpReadVariableOpconv_3_9/kernel*'
_output_shapes
:@А*
dtype0
s
conv_3_9/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*
shared_nameconv_3_9/bias
l
!conv_3_9/bias/Read/ReadVariableOpReadVariableOpconv_3_9/bias*
_output_shapes	
:А*
dtype0
y
b_norm_3_9/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*!
shared_nameb_norm_3_9/gamma
r
$b_norm_3_9/gamma/Read/ReadVariableOpReadVariableOpb_norm_3_9/gamma*
_output_shapes	
:А*
dtype0
w
b_norm_3_9/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:А* 
shared_nameb_norm_3_9/beta
p
#b_norm_3_9/beta/Read/ReadVariableOpReadVariableOpb_norm_3_9/beta*
_output_shapes	
:А*
dtype0
Е
b_norm_3_9/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*'
shared_nameb_norm_3_9/moving_mean
~
*b_norm_3_9/moving_mean/Read/ReadVariableOpReadVariableOpb_norm_3_9/moving_mean*
_output_shapes	
:А*
dtype0
Н
b_norm_3_9/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*+
shared_nameb_norm_3_9/moving_variance
Ж
.b_norm_3_9/moving_variance/Read/ReadVariableOpReadVariableOpb_norm_3_9/moving_variance*
_output_shapes	
:А*
dtype0
Д
conv_4_4/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:АА* 
shared_nameconv_4_4/kernel
}
#conv_4_4/kernel/Read/ReadVariableOpReadVariableOpconv_4_4/kernel*(
_output_shapes
:АА*
dtype0
s
conv_4_4/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*
shared_nameconv_4_4/bias
l
!conv_4_4/bias/Read/ReadVariableOpReadVariableOpconv_4_4/bias*
_output_shapes	
:А*
dtype0
y
b_norm_4_4/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*!
shared_nameb_norm_4_4/gamma
r
$b_norm_4_4/gamma/Read/ReadVariableOpReadVariableOpb_norm_4_4/gamma*
_output_shapes	
:А*
dtype0
w
b_norm_4_4/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:А* 
shared_nameb_norm_4_4/beta
p
#b_norm_4_4/beta/Read/ReadVariableOpReadVariableOpb_norm_4_4/beta*
_output_shapes	
:А*
dtype0
Е
b_norm_4_4/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*'
shared_nameb_norm_4_4/moving_mean
~
*b_norm_4_4/moving_mean/Read/ReadVariableOpReadVariableOpb_norm_4_4/moving_mean*
_output_shapes	
:А*
dtype0
Н
b_norm_4_4/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*+
shared_nameb_norm_4_4/moving_variance
Ж
.b_norm_4_4/moving_variance/Read/ReadVariableOpReadVariableOpb_norm_4_4/moving_variance*
_output_shapes	
:А*
dtype0
д
#output_class_distribution_14/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
А╚*4
shared_name%#output_class_distribution_14/kernel
Э
7output_class_distribution_14/kernel/Read/ReadVariableOpReadVariableOp#output_class_distribution_14/kernel* 
_output_shapes
:
А╚*
dtype0
Ы
!output_class_distribution_14/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:╚*2
shared_name#!output_class_distribution_14/bias
Ф
5output_class_distribution_14/bias/Read/ReadVariableOpReadVariableOp!output_class_distribution_14/bias*
_output_shapes	
:╚*
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
Ъ
Adam/input_conv_14/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *,
shared_nameAdam/input_conv_14/kernel/m
У
/Adam/input_conv_14/kernel/m/Read/ReadVariableOpReadVariableOpAdam/input_conv_14/kernel/m*&
_output_shapes
: *
dtype0
К
Adam/input_conv_14/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: **
shared_nameAdam/input_conv_14/bias/m
Г
-Adam/input_conv_14/bias/m/Read/ReadVariableOpReadVariableOpAdam/input_conv_14/bias/m*
_output_shapes
: *
dtype0
к
)Adam/input_batch_normalization_14/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *:
shared_name+)Adam/input_batch_normalization_14/gamma/m
г
=Adam/input_batch_normalization_14/gamma/m/Read/ReadVariableOpReadVariableOp)Adam/input_batch_normalization_14/gamma/m*
_output_shapes
: *
dtype0
и
(Adam/input_batch_normalization_14/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *9
shared_name*(Adam/input_batch_normalization_14/beta/m
б
<Adam/input_batch_normalization_14/beta/m/Read/ReadVariableOpReadVariableOp(Adam/input_batch_normalization_14/beta/m*
_output_shapes
: *
dtype0
Т
Adam/conv_1_14/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:  *(
shared_nameAdam/conv_1_14/kernel/m
Л
+Adam/conv_1_14/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv_1_14/kernel/m*&
_output_shapes
:  *
dtype0
В
Adam/conv_1_14/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *&
shared_nameAdam/conv_1_14/bias/m
{
)Adam/conv_1_14/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv_1_14/bias/m*
_output_shapes
: *
dtype0
И
Adam/b_norm_1_14/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *)
shared_nameAdam/b_norm_1_14/gamma/m
Б
,Adam/b_norm_1_14/gamma/m/Read/ReadVariableOpReadVariableOpAdam/b_norm_1_14/gamma/m*
_output_shapes
: *
dtype0
Ж
Adam/b_norm_1_14/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *(
shared_nameAdam/b_norm_1_14/beta/m

+Adam/b_norm_1_14/beta/m/Read/ReadVariableOpReadVariableOpAdam/b_norm_1_14/beta/m*
_output_shapes
: *
dtype0
Т
Adam/conv_2_14/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: @*(
shared_nameAdam/conv_2_14/kernel/m
Л
+Adam/conv_2_14/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv_2_14/kernel/m*&
_output_shapes
: @*
dtype0
В
Adam/conv_2_14/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*&
shared_nameAdam/conv_2_14/bias/m
{
)Adam/conv_2_14/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv_2_14/bias/m*
_output_shapes
:@*
dtype0
И
Adam/b_norm_2_14/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*)
shared_nameAdam/b_norm_2_14/gamma/m
Б
,Adam/b_norm_2_14/gamma/m/Read/ReadVariableOpReadVariableOpAdam/b_norm_2_14/gamma/m*
_output_shapes
:@*
dtype0
Ж
Adam/b_norm_2_14/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*(
shared_nameAdam/b_norm_2_14/beta/m

+Adam/b_norm_2_14/beta/m/Read/ReadVariableOpReadVariableOpAdam/b_norm_2_14/beta/m*
_output_shapes
:@*
dtype0
С
Adam/conv_3_9/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@А*'
shared_nameAdam/conv_3_9/kernel/m
К
*Adam/conv_3_9/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv_3_9/kernel/m*'
_output_shapes
:@А*
dtype0
Б
Adam/conv_3_9/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*%
shared_nameAdam/conv_3_9/bias/m
z
(Adam/conv_3_9/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv_3_9/bias/m*
_output_shapes	
:А*
dtype0
З
Adam/b_norm_3_9/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*(
shared_nameAdam/b_norm_3_9/gamma/m
А
+Adam/b_norm_3_9/gamma/m/Read/ReadVariableOpReadVariableOpAdam/b_norm_3_9/gamma/m*
_output_shapes	
:А*
dtype0
Е
Adam/b_norm_3_9/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*'
shared_nameAdam/b_norm_3_9/beta/m
~
*Adam/b_norm_3_9/beta/m/Read/ReadVariableOpReadVariableOpAdam/b_norm_3_9/beta/m*
_output_shapes	
:А*
dtype0
Т
Adam/conv_4_4/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:АА*'
shared_nameAdam/conv_4_4/kernel/m
Л
*Adam/conv_4_4/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv_4_4/kernel/m*(
_output_shapes
:АА*
dtype0
Б
Adam/conv_4_4/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*%
shared_nameAdam/conv_4_4/bias/m
z
(Adam/conv_4_4/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv_4_4/bias/m*
_output_shapes	
:А*
dtype0
З
Adam/b_norm_4_4/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*(
shared_nameAdam/b_norm_4_4/gamma/m
А
+Adam/b_norm_4_4/gamma/m/Read/ReadVariableOpReadVariableOpAdam/b_norm_4_4/gamma/m*
_output_shapes	
:А*
dtype0
Е
Adam/b_norm_4_4/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*'
shared_nameAdam/b_norm_4_4/beta/m
~
*Adam/b_norm_4_4/beta/m/Read/ReadVariableOpReadVariableOpAdam/b_norm_4_4/beta/m*
_output_shapes	
:А*
dtype0
▓
*Adam/output_class_distribution_14/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
А╚*;
shared_name,*Adam/output_class_distribution_14/kernel/m
л
>Adam/output_class_distribution_14/kernel/m/Read/ReadVariableOpReadVariableOp*Adam/output_class_distribution_14/kernel/m* 
_output_shapes
:
А╚*
dtype0
й
(Adam/output_class_distribution_14/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:╚*9
shared_name*(Adam/output_class_distribution_14/bias/m
в
<Adam/output_class_distribution_14/bias/m/Read/ReadVariableOpReadVariableOp(Adam/output_class_distribution_14/bias/m*
_output_shapes	
:╚*
dtype0
Ъ
Adam/input_conv_14/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *,
shared_nameAdam/input_conv_14/kernel/v
У
/Adam/input_conv_14/kernel/v/Read/ReadVariableOpReadVariableOpAdam/input_conv_14/kernel/v*&
_output_shapes
: *
dtype0
К
Adam/input_conv_14/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: **
shared_nameAdam/input_conv_14/bias/v
Г
-Adam/input_conv_14/bias/v/Read/ReadVariableOpReadVariableOpAdam/input_conv_14/bias/v*
_output_shapes
: *
dtype0
к
)Adam/input_batch_normalization_14/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *:
shared_name+)Adam/input_batch_normalization_14/gamma/v
г
=Adam/input_batch_normalization_14/gamma/v/Read/ReadVariableOpReadVariableOp)Adam/input_batch_normalization_14/gamma/v*
_output_shapes
: *
dtype0
и
(Adam/input_batch_normalization_14/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *9
shared_name*(Adam/input_batch_normalization_14/beta/v
б
<Adam/input_batch_normalization_14/beta/v/Read/ReadVariableOpReadVariableOp(Adam/input_batch_normalization_14/beta/v*
_output_shapes
: *
dtype0
Т
Adam/conv_1_14/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:  *(
shared_nameAdam/conv_1_14/kernel/v
Л
+Adam/conv_1_14/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv_1_14/kernel/v*&
_output_shapes
:  *
dtype0
В
Adam/conv_1_14/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *&
shared_nameAdam/conv_1_14/bias/v
{
)Adam/conv_1_14/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv_1_14/bias/v*
_output_shapes
: *
dtype0
И
Adam/b_norm_1_14/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *)
shared_nameAdam/b_norm_1_14/gamma/v
Б
,Adam/b_norm_1_14/gamma/v/Read/ReadVariableOpReadVariableOpAdam/b_norm_1_14/gamma/v*
_output_shapes
: *
dtype0
Ж
Adam/b_norm_1_14/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *(
shared_nameAdam/b_norm_1_14/beta/v

+Adam/b_norm_1_14/beta/v/Read/ReadVariableOpReadVariableOpAdam/b_norm_1_14/beta/v*
_output_shapes
: *
dtype0
Т
Adam/conv_2_14/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: @*(
shared_nameAdam/conv_2_14/kernel/v
Л
+Adam/conv_2_14/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv_2_14/kernel/v*&
_output_shapes
: @*
dtype0
В
Adam/conv_2_14/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*&
shared_nameAdam/conv_2_14/bias/v
{
)Adam/conv_2_14/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv_2_14/bias/v*
_output_shapes
:@*
dtype0
И
Adam/b_norm_2_14/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*)
shared_nameAdam/b_norm_2_14/gamma/v
Б
,Adam/b_norm_2_14/gamma/v/Read/ReadVariableOpReadVariableOpAdam/b_norm_2_14/gamma/v*
_output_shapes
:@*
dtype0
Ж
Adam/b_norm_2_14/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*(
shared_nameAdam/b_norm_2_14/beta/v

+Adam/b_norm_2_14/beta/v/Read/ReadVariableOpReadVariableOpAdam/b_norm_2_14/beta/v*
_output_shapes
:@*
dtype0
С
Adam/conv_3_9/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@А*'
shared_nameAdam/conv_3_9/kernel/v
К
*Adam/conv_3_9/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv_3_9/kernel/v*'
_output_shapes
:@А*
dtype0
Б
Adam/conv_3_9/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*%
shared_nameAdam/conv_3_9/bias/v
z
(Adam/conv_3_9/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv_3_9/bias/v*
_output_shapes	
:А*
dtype0
З
Adam/b_norm_3_9/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*(
shared_nameAdam/b_norm_3_9/gamma/v
А
+Adam/b_norm_3_9/gamma/v/Read/ReadVariableOpReadVariableOpAdam/b_norm_3_9/gamma/v*
_output_shapes	
:А*
dtype0
Е
Adam/b_norm_3_9/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*'
shared_nameAdam/b_norm_3_9/beta/v
~
*Adam/b_norm_3_9/beta/v/Read/ReadVariableOpReadVariableOpAdam/b_norm_3_9/beta/v*
_output_shapes	
:А*
dtype0
Т
Adam/conv_4_4/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:АА*'
shared_nameAdam/conv_4_4/kernel/v
Л
*Adam/conv_4_4/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv_4_4/kernel/v*(
_output_shapes
:АА*
dtype0
Б
Adam/conv_4_4/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*%
shared_nameAdam/conv_4_4/bias/v
z
(Adam/conv_4_4/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv_4_4/bias/v*
_output_shapes	
:А*
dtype0
З
Adam/b_norm_4_4/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*(
shared_nameAdam/b_norm_4_4/gamma/v
А
+Adam/b_norm_4_4/gamma/v/Read/ReadVariableOpReadVariableOpAdam/b_norm_4_4/gamma/v*
_output_shapes	
:А*
dtype0
Е
Adam/b_norm_4_4/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*'
shared_nameAdam/b_norm_4_4/beta/v
~
*Adam/b_norm_4_4/beta/v/Read/ReadVariableOpReadVariableOpAdam/b_norm_4_4/beta/v*
_output_shapes	
:А*
dtype0
▓
*Adam/output_class_distribution_14/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
А╚*;
shared_name,*Adam/output_class_distribution_14/kernel/v
л
>Adam/output_class_distribution_14/kernel/v/Read/ReadVariableOpReadVariableOp*Adam/output_class_distribution_14/kernel/v* 
_output_shapes
:
А╚*
dtype0
й
(Adam/output_class_distribution_14/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:╚*9
shared_name*(Adam/output_class_distribution_14/bias/v
в
<Adam/output_class_distribution_14/bias/v/Read/ReadVariableOpReadVariableOp(Adam/output_class_distribution_14/bias/v*
_output_shapes	
:╚*
dtype0

NoOpNoOp
░Ч
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*ъЦ
value▀ЦB█Ц B╙Ц
ф
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
	variables
regularization_losses
trainable_variables
	keras_api

signatures
 
h

 kernel
!bias
"	variables
#regularization_losses
$trainable_variables
%	keras_api
Ч
&axis
	'gamma
(beta
)moving_mean
*moving_variance
+	variables
,regularization_losses
-trainable_variables
.	keras_api
R
/	variables
0regularization_losses
1trainable_variables
2	keras_api
R
3	variables
4regularization_losses
5trainable_variables
6	keras_api
R
7	variables
8regularization_losses
9trainable_variables
:	keras_api
h

;kernel
<bias
=	variables
>regularization_losses
?trainable_variables
@	keras_api
Ч
Aaxis
	Bgamma
Cbeta
Dmoving_mean
Emoving_variance
F	variables
Gregularization_losses
Htrainable_variables
I	keras_api
R
J	variables
Kregularization_losses
Ltrainable_variables
M	keras_api
R
N	variables
Oregularization_losses
Ptrainable_variables
Q	keras_api
h

Rkernel
Sbias
T	variables
Uregularization_losses
Vtrainable_variables
W	keras_api
Ч
Xaxis
	Ygamma
Zbeta
[moving_mean
\moving_variance
]	variables
^regularization_losses
_trainable_variables
`	keras_api
R
a	variables
bregularization_losses
ctrainable_variables
d	keras_api
R
e	variables
fregularization_losses
gtrainable_variables
h	keras_api
h

ikernel
jbias
k	variables
lregularization_losses
mtrainable_variables
n	keras_api
Ч
oaxis
	pgamma
qbeta
rmoving_mean
smoving_variance
t	variables
uregularization_losses
vtrainable_variables
w	keras_api
R
x	variables
yregularization_losses
ztrainable_variables
{	keras_api
R
|	variables
}regularization_losses
~trainable_variables
	keras_api
n
Аkernel
	Бbias
В	variables
Гregularization_losses
Дtrainable_variables
Е	keras_api
а
	Жaxis

Зgamma
	Иbeta
Йmoving_mean
Кmoving_variance
Л	variables
Мregularization_losses
Нtrainable_variables
О	keras_api
V
П	variables
Рregularization_losses
Сtrainable_variables
Т	keras_api
V
У	variables
Фregularization_losses
Хtrainable_variables
Ц	keras_api
V
Ч	variables
Шregularization_losses
Щtrainable_variables
Ъ	keras_api
V
Ы	variables
Ьregularization_losses
Эtrainable_variables
Ю	keras_api
n
Яkernel
	аbias
б	variables
вregularization_losses
гtrainable_variables
д	keras_api
Й
	еiter
жbeta_1
зbeta_2

иdecay
йlearning_rate mЪ!mЫ'mЬ(mЭ;mЮ<mЯBmаCmбRmвSmгYmдZmеimжjmзpmиqmй	Аmк	Бmл	Зmм	Иmн	Яmо	аmп v░!v▒'v▓(v│;v┤<v╡Bv╢Cv╖Rv╕Sv╣Yv║Zv╗iv╝jv╜pv╛qv┐	Аv└	Бv┴	Зv┬	Иv├	Яv─	аv┼
■
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
А24
Б25
З26
И27
Й28
К29
Я30
а31
 
м
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
А16
Б17
З18
И19
Я20
а21
Ю
кlayers
 лlayer_regularization_losses
мmetrics
нnon_trainable_variables
	variables
regularization_losses
trainable_variables
 
`^
VARIABLE_VALUEinput_conv_14/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE
\Z
VARIABLE_VALUEinput_conv_14/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE

 0
!1
 

 0
!1
Ю
оlayers
 пlayer_regularization_losses
░metrics
▒non_trainable_variables
"	variables
#regularization_losses
$trainable_variables
 
mk
VARIABLE_VALUE"input_batch_normalization_14/gamma5layer_with_weights-1/gamma/.ATTRIBUTES/VARIABLE_VALUE
ki
VARIABLE_VALUE!input_batch_normalization_14/beta4layer_with_weights-1/beta/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUE(input_batch_normalization_14/moving_mean;layer_with_weights-1/moving_mean/.ATTRIBUTES/VARIABLE_VALUE
Б
VARIABLE_VALUE,input_batch_normalization_14/moving_variance?layer_with_weights-1/moving_variance/.ATTRIBUTES/VARIABLE_VALUE

'0
(1
)2
*3
 

'0
(1
Ю
▓layers
 │layer_regularization_losses
┤metrics
╡non_trainable_variables
+	variables
,regularization_losses
-trainable_variables
 
 
 
Ю
╢layers
 ╖layer_regularization_losses
╕metrics
╣non_trainable_variables
/	variables
0regularization_losses
1trainable_variables
 
 
 
Ю
║layers
 ╗layer_regularization_losses
╝metrics
╜non_trainable_variables
3	variables
4regularization_losses
5trainable_variables
 
 
 
Ю
╛layers
 ┐layer_regularization_losses
└metrics
┴non_trainable_variables
7	variables
8regularization_losses
9trainable_variables
\Z
VARIABLE_VALUEconv_1_14/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEconv_1_14/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE

;0
<1
 

;0
<1
Ю
┬layers
 ├layer_regularization_losses
─metrics
┼non_trainable_variables
=	variables
>regularization_losses
?trainable_variables
 
\Z
VARIABLE_VALUEb_norm_1_14/gamma5layer_with_weights-3/gamma/.ATTRIBUTES/VARIABLE_VALUE
ZX
VARIABLE_VALUEb_norm_1_14/beta4layer_with_weights-3/beta/.ATTRIBUTES/VARIABLE_VALUE
hf
VARIABLE_VALUEb_norm_1_14/moving_mean;layer_with_weights-3/moving_mean/.ATTRIBUTES/VARIABLE_VALUE
pn
VARIABLE_VALUEb_norm_1_14/moving_variance?layer_with_weights-3/moving_variance/.ATTRIBUTES/VARIABLE_VALUE

B0
C1
D2
E3
 

B0
C1
Ю
╞layers
 ╟layer_regularization_losses
╚metrics
╔non_trainable_variables
F	variables
Gregularization_losses
Htrainable_variables
 
 
 
Ю
╩layers
 ╦layer_regularization_losses
╠metrics
═non_trainable_variables
J	variables
Kregularization_losses
Ltrainable_variables
 
 
 
Ю
╬layers
 ╧layer_regularization_losses
╨metrics
╤non_trainable_variables
N	variables
Oregularization_losses
Ptrainable_variables
\Z
VARIABLE_VALUEconv_2_14/kernel6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEconv_2_14/bias4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUE

R0
S1
 

R0
S1
Ю
╥layers
 ╙layer_regularization_losses
╘metrics
╒non_trainable_variables
T	variables
Uregularization_losses
Vtrainable_variables
 
\Z
VARIABLE_VALUEb_norm_2_14/gamma5layer_with_weights-5/gamma/.ATTRIBUTES/VARIABLE_VALUE
ZX
VARIABLE_VALUEb_norm_2_14/beta4layer_with_weights-5/beta/.ATTRIBUTES/VARIABLE_VALUE
hf
VARIABLE_VALUEb_norm_2_14/moving_mean;layer_with_weights-5/moving_mean/.ATTRIBUTES/VARIABLE_VALUE
pn
VARIABLE_VALUEb_norm_2_14/moving_variance?layer_with_weights-5/moving_variance/.ATTRIBUTES/VARIABLE_VALUE

Y0
Z1
[2
\3
 

Y0
Z1
Ю
╓layers
 ╫layer_regularization_losses
╪metrics
┘non_trainable_variables
]	variables
^regularization_losses
_trainable_variables
 
 
 
Ю
┌layers
 █layer_regularization_losses
▄metrics
▌non_trainable_variables
a	variables
bregularization_losses
ctrainable_variables
 
 
 
Ю
▐layers
 ▀layer_regularization_losses
рmetrics
сnon_trainable_variables
e	variables
fregularization_losses
gtrainable_variables
[Y
VARIABLE_VALUEconv_3_9/kernel6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEconv_3_9/bias4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUE

i0
j1
 

i0
j1
Ю
тlayers
 уlayer_regularization_losses
фmetrics
хnon_trainable_variables
k	variables
lregularization_losses
mtrainable_variables
 
[Y
VARIABLE_VALUEb_norm_3_9/gamma5layer_with_weights-7/gamma/.ATTRIBUTES/VARIABLE_VALUE
YW
VARIABLE_VALUEb_norm_3_9/beta4layer_with_weights-7/beta/.ATTRIBUTES/VARIABLE_VALUE
ge
VARIABLE_VALUEb_norm_3_9/moving_mean;layer_with_weights-7/moving_mean/.ATTRIBUTES/VARIABLE_VALUE
om
VARIABLE_VALUEb_norm_3_9/moving_variance?layer_with_weights-7/moving_variance/.ATTRIBUTES/VARIABLE_VALUE

p0
q1
r2
s3
 

p0
q1
Ю
цlayers
 чlayer_regularization_losses
шmetrics
щnon_trainable_variables
t	variables
uregularization_losses
vtrainable_variables
 
 
 
Ю
ъlayers
 ыlayer_regularization_losses
ьmetrics
эnon_trainable_variables
x	variables
yregularization_losses
ztrainable_variables
 
 
 
Ю
юlayers
 яlayer_regularization_losses
Ёmetrics
ёnon_trainable_variables
|	variables
}regularization_losses
~trainable_variables
[Y
VARIABLE_VALUEconv_4_4/kernel6layer_with_weights-8/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEconv_4_4/bias4layer_with_weights-8/bias/.ATTRIBUTES/VARIABLE_VALUE

А0
Б1
 

А0
Б1
б
Єlayers
 єlayer_regularization_losses
Їmetrics
їnon_trainable_variables
В	variables
Гregularization_losses
Дtrainable_variables
 
[Y
VARIABLE_VALUEb_norm_4_4/gamma5layer_with_weights-9/gamma/.ATTRIBUTES/VARIABLE_VALUE
YW
VARIABLE_VALUEb_norm_4_4/beta4layer_with_weights-9/beta/.ATTRIBUTES/VARIABLE_VALUE
ge
VARIABLE_VALUEb_norm_4_4/moving_mean;layer_with_weights-9/moving_mean/.ATTRIBUTES/VARIABLE_VALUE
om
VARIABLE_VALUEb_norm_4_4/moving_variance?layer_with_weights-9/moving_variance/.ATTRIBUTES/VARIABLE_VALUE
 
З0
И1
Й2
К3
 

З0
И1
б
Ўlayers
 ўlayer_regularization_losses
°metrics
∙non_trainable_variables
Л	variables
Мregularization_losses
Нtrainable_variables
 
 
 
б
·layers
 √layer_regularization_losses
№metrics
¤non_trainable_variables
П	variables
Рregularization_losses
Сtrainable_variables
 
 
 
б
■layers
  layer_regularization_losses
Аmetrics
Бnon_trainable_variables
У	variables
Фregularization_losses
Хtrainable_variables
 
 
 
б
Вlayers
 Гlayer_regularization_losses
Дmetrics
Еnon_trainable_variables
Ч	variables
Шregularization_losses
Щtrainable_variables
 
 
 
б
Жlayers
 Зlayer_regularization_losses
Иmetrics
Йnon_trainable_variables
Ы	variables
Ьregularization_losses
Эtrainable_variables
pn
VARIABLE_VALUE#output_class_distribution_14/kernel7layer_with_weights-10/kernel/.ATTRIBUTES/VARIABLE_VALUE
lj
VARIABLE_VALUE!output_class_distribution_14/bias5layer_with_weights-10/bias/.ATTRIBUTES/VARIABLE_VALUE

Я0
а1
 

Я0
а1
б
Кlayers
 Лlayer_regularization_losses
Мmetrics
Нnon_trainable_variables
б	variables
вregularization_losses
гtrainable_variables
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
╢
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
 

О0
H
)0
*1
D2
E3
[4
\5
r6
s7
Й8
К9
 
 
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
Й0
К1
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

Пtotal

Рcount
С
_fn_kwargs
Т	variables
Уregularization_losses
Фtrainable_variables
Х	keras_api
OM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE
OM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE
 

П0
Р1
 
 
б
Цlayers
 Чlayer_regularization_losses
Шmetrics
Щnon_trainable_variables
Т	variables
Уregularization_losses
Фtrainable_variables
 
 
 

П0
Р1
ДБ
VARIABLE_VALUEAdam/input_conv_14/kernel/mRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/input_conv_14/bias/mPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
СО
VARIABLE_VALUE)Adam/input_batch_normalization_14/gamma/mQlayer_with_weights-1/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
ПМ
VARIABLE_VALUE(Adam/input_batch_normalization_14/beta/mPlayer_with_weights-1/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/conv_1_14/kernel/mRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/conv_1_14/bias/mPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/b_norm_1_14/gamma/mQlayer_with_weights-3/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}{
VARIABLE_VALUEAdam/b_norm_1_14/beta/mPlayer_with_weights-3/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/conv_2_14/kernel/mRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/conv_2_14/bias/mPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/b_norm_2_14/gamma/mQlayer_with_weights-5/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}{
VARIABLE_VALUEAdam/b_norm_2_14/beta/mPlayer_with_weights-5/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/conv_3_9/kernel/mRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/conv_3_9/bias/mPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/b_norm_3_9/gamma/mQlayer_with_weights-7/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUEAdam/b_norm_3_9/beta/mPlayer_with_weights-7/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/conv_4_4/kernel/mRlayer_with_weights-8/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/conv_4_4/bias/mPlayer_with_weights-8/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/b_norm_4_4/gamma/mQlayer_with_weights-9/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUEAdam/b_norm_4_4/beta/mPlayer_with_weights-9/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
ФС
VARIABLE_VALUE*Adam/output_class_distribution_14/kernel/mSlayer_with_weights-10/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
РН
VARIABLE_VALUE(Adam/output_class_distribution_14/bias/mQlayer_with_weights-10/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
ДБ
VARIABLE_VALUEAdam/input_conv_14/kernel/vRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/input_conv_14/bias/vPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
СО
VARIABLE_VALUE)Adam/input_batch_normalization_14/gamma/vQlayer_with_weights-1/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
ПМ
VARIABLE_VALUE(Adam/input_batch_normalization_14/beta/vPlayer_with_weights-1/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/conv_1_14/kernel/vRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/conv_1_14/bias/vPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/b_norm_1_14/gamma/vQlayer_with_weights-3/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}{
VARIABLE_VALUEAdam/b_norm_1_14/beta/vPlayer_with_weights-3/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/conv_2_14/kernel/vRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/conv_2_14/bias/vPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/b_norm_2_14/gamma/vQlayer_with_weights-5/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}{
VARIABLE_VALUEAdam/b_norm_2_14/beta/vPlayer_with_weights-5/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/conv_3_9/kernel/vRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/conv_3_9/bias/vPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/b_norm_3_9/gamma/vQlayer_with_weights-7/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUEAdam/b_norm_3_9/beta/vPlayer_with_weights-7/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/conv_4_4/kernel/vRlayer_with_weights-8/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/conv_4_4/bias/vPlayer_with_weights-8/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/b_norm_4_4/gamma/vQlayer_with_weights-9/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUEAdam/b_norm_4_4/beta/vPlayer_with_weights-9/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
ФС
VARIABLE_VALUE*Adam/output_class_distribution_14/kernel/vSlayer_with_weights-10/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
РН
VARIABLE_VALUE(Adam/output_class_distribution_14/bias/vQlayer_with_weights-10/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
Ч
 serving_default_input_conv_inputPlaceholder*1
_output_shapes
:         АА*
dtype0*&
shape:         АА
┴
StatefulPartitionedCallStatefulPartitionedCall serving_default_input_conv_inputinput_conv_14/kernelinput_conv_14/bias"input_batch_normalization_14/gamma!input_batch_normalization_14/beta(input_batch_normalization_14/moving_mean,input_batch_normalization_14/moving_varianceconv_1_14/kernelconv_1_14/biasb_norm_1_14/gammab_norm_1_14/betab_norm_1_14/moving_meanb_norm_1_14/moving_varianceconv_2_14/kernelconv_2_14/biasb_norm_2_14/gammab_norm_2_14/betab_norm_2_14/moving_meanb_norm_2_14/moving_varianceconv_3_9/kernelconv_3_9/biasb_norm_3_9/gammab_norm_3_9/betab_norm_3_9/moving_meanb_norm_3_9/moving_varianceconv_4_4/kernelconv_4_4/biasb_norm_4_4/gammab_norm_4_4/betab_norm_4_4/moving_meanb_norm_4_4/moving_variance#output_class_distribution_14/kernel!output_class_distribution_14/bias*,
Tin%
#2!*
Tout
2*,
_gradient_op_typePartitionedCallUnused*(
_output_shapes
:         ╚*-
config_proto

GPU

CPU2*0J 8*.
f)R'
%__inference_signature_wrapper_2737076
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
Є
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename(input_conv_14/kernel/Read/ReadVariableOp&input_conv_14/bias/Read/ReadVariableOp6input_batch_normalization_14/gamma/Read/ReadVariableOp5input_batch_normalization_14/beta/Read/ReadVariableOp<input_batch_normalization_14/moving_mean/Read/ReadVariableOp@input_batch_normalization_14/moving_variance/Read/ReadVariableOp$conv_1_14/kernel/Read/ReadVariableOp"conv_1_14/bias/Read/ReadVariableOp%b_norm_1_14/gamma/Read/ReadVariableOp$b_norm_1_14/beta/Read/ReadVariableOp+b_norm_1_14/moving_mean/Read/ReadVariableOp/b_norm_1_14/moving_variance/Read/ReadVariableOp$conv_2_14/kernel/Read/ReadVariableOp"conv_2_14/bias/Read/ReadVariableOp%b_norm_2_14/gamma/Read/ReadVariableOp$b_norm_2_14/beta/Read/ReadVariableOp+b_norm_2_14/moving_mean/Read/ReadVariableOp/b_norm_2_14/moving_variance/Read/ReadVariableOp#conv_3_9/kernel/Read/ReadVariableOp!conv_3_9/bias/Read/ReadVariableOp$b_norm_3_9/gamma/Read/ReadVariableOp#b_norm_3_9/beta/Read/ReadVariableOp*b_norm_3_9/moving_mean/Read/ReadVariableOp.b_norm_3_9/moving_variance/Read/ReadVariableOp#conv_4_4/kernel/Read/ReadVariableOp!conv_4_4/bias/Read/ReadVariableOp$b_norm_4_4/gamma/Read/ReadVariableOp#b_norm_4_4/beta/Read/ReadVariableOp*b_norm_4_4/moving_mean/Read/ReadVariableOp.b_norm_4_4/moving_variance/Read/ReadVariableOp7output_class_distribution_14/kernel/Read/ReadVariableOp5output_class_distribution_14/bias/Read/ReadVariableOpAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOp/Adam/input_conv_14/kernel/m/Read/ReadVariableOp-Adam/input_conv_14/bias/m/Read/ReadVariableOp=Adam/input_batch_normalization_14/gamma/m/Read/ReadVariableOp<Adam/input_batch_normalization_14/beta/m/Read/ReadVariableOp+Adam/conv_1_14/kernel/m/Read/ReadVariableOp)Adam/conv_1_14/bias/m/Read/ReadVariableOp,Adam/b_norm_1_14/gamma/m/Read/ReadVariableOp+Adam/b_norm_1_14/beta/m/Read/ReadVariableOp+Adam/conv_2_14/kernel/m/Read/ReadVariableOp)Adam/conv_2_14/bias/m/Read/ReadVariableOp,Adam/b_norm_2_14/gamma/m/Read/ReadVariableOp+Adam/b_norm_2_14/beta/m/Read/ReadVariableOp*Adam/conv_3_9/kernel/m/Read/ReadVariableOp(Adam/conv_3_9/bias/m/Read/ReadVariableOp+Adam/b_norm_3_9/gamma/m/Read/ReadVariableOp*Adam/b_norm_3_9/beta/m/Read/ReadVariableOp*Adam/conv_4_4/kernel/m/Read/ReadVariableOp(Adam/conv_4_4/bias/m/Read/ReadVariableOp+Adam/b_norm_4_4/gamma/m/Read/ReadVariableOp*Adam/b_norm_4_4/beta/m/Read/ReadVariableOp>Adam/output_class_distribution_14/kernel/m/Read/ReadVariableOp<Adam/output_class_distribution_14/bias/m/Read/ReadVariableOp/Adam/input_conv_14/kernel/v/Read/ReadVariableOp-Adam/input_conv_14/bias/v/Read/ReadVariableOp=Adam/input_batch_normalization_14/gamma/v/Read/ReadVariableOp<Adam/input_batch_normalization_14/beta/v/Read/ReadVariableOp+Adam/conv_1_14/kernel/v/Read/ReadVariableOp)Adam/conv_1_14/bias/v/Read/ReadVariableOp,Adam/b_norm_1_14/gamma/v/Read/ReadVariableOp+Adam/b_norm_1_14/beta/v/Read/ReadVariableOp+Adam/conv_2_14/kernel/v/Read/ReadVariableOp)Adam/conv_2_14/bias/v/Read/ReadVariableOp,Adam/b_norm_2_14/gamma/v/Read/ReadVariableOp+Adam/b_norm_2_14/beta/v/Read/ReadVariableOp*Adam/conv_3_9/kernel/v/Read/ReadVariableOp(Adam/conv_3_9/bias/v/Read/ReadVariableOp+Adam/b_norm_3_9/gamma/v/Read/ReadVariableOp*Adam/b_norm_3_9/beta/v/Read/ReadVariableOp*Adam/conv_4_4/kernel/v/Read/ReadVariableOp(Adam/conv_4_4/bias/v/Read/ReadVariableOp+Adam/b_norm_4_4/gamma/v/Read/ReadVariableOp*Adam/b_norm_4_4/beta/v/Read/ReadVariableOp>Adam/output_class_distribution_14/kernel/v/Read/ReadVariableOp<Adam/output_class_distribution_14/bias/v/Read/ReadVariableOpConst*`
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
 __inference__traced_save_2738749
ё
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenameinput_conv_14/kernelinput_conv_14/bias"input_batch_normalization_14/gamma!input_batch_normalization_14/beta(input_batch_normalization_14/moving_mean,input_batch_normalization_14/moving_varianceconv_1_14/kernelconv_1_14/biasb_norm_1_14/gammab_norm_1_14/betab_norm_1_14/moving_meanb_norm_1_14/moving_varianceconv_2_14/kernelconv_2_14/biasb_norm_2_14/gammab_norm_2_14/betab_norm_2_14/moving_meanb_norm_2_14/moving_varianceconv_3_9/kernelconv_3_9/biasb_norm_3_9/gammab_norm_3_9/betab_norm_3_9/moving_meanb_norm_3_9/moving_varianceconv_4_4/kernelconv_4_4/biasb_norm_4_4/gammab_norm_4_4/betab_norm_4_4/moving_meanb_norm_4_4/moving_variance#output_class_distribution_14/kernel!output_class_distribution_14/bias	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_ratetotalcountAdam/input_conv_14/kernel/mAdam/input_conv_14/bias/m)Adam/input_batch_normalization_14/gamma/m(Adam/input_batch_normalization_14/beta/mAdam/conv_1_14/kernel/mAdam/conv_1_14/bias/mAdam/b_norm_1_14/gamma/mAdam/b_norm_1_14/beta/mAdam/conv_2_14/kernel/mAdam/conv_2_14/bias/mAdam/b_norm_2_14/gamma/mAdam/b_norm_2_14/beta/mAdam/conv_3_9/kernel/mAdam/conv_3_9/bias/mAdam/b_norm_3_9/gamma/mAdam/b_norm_3_9/beta/mAdam/conv_4_4/kernel/mAdam/conv_4_4/bias/mAdam/b_norm_4_4/gamma/mAdam/b_norm_4_4/beta/m*Adam/output_class_distribution_14/kernel/m(Adam/output_class_distribution_14/bias/mAdam/input_conv_14/kernel/vAdam/input_conv_14/bias/v)Adam/input_batch_normalization_14/gamma/v(Adam/input_batch_normalization_14/beta/vAdam/conv_1_14/kernel/vAdam/conv_1_14/bias/vAdam/b_norm_1_14/gamma/vAdam/b_norm_1_14/beta/vAdam/conv_2_14/kernel/vAdam/conv_2_14/bias/vAdam/b_norm_2_14/gamma/vAdam/b_norm_2_14/beta/vAdam/conv_3_9/kernel/vAdam/conv_3_9/bias/vAdam/b_norm_3_9/gamma/vAdam/b_norm_3_9/beta/vAdam/conv_4_4/kernel/vAdam/conv_4_4/bias/vAdam/b_norm_4_4/gamma/vAdam/b_norm_4_4/beta/v*Adam/output_class_distribution_14/kernel/v(Adam/output_class_distribution_14/bias/v*_
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
#__inference__traced_restore_2739010└╣
╥
N
2__inference_max_pooling2d_59_layer_call_fn_2736118

inputs
identity█
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*J
_output_shapes8
6:4                                    *-
config_proto

GPU

CPU2*0J 8*V
fQRO
M__inference_max_pooling2d_59_layer_call_and_return_conditional_losses_27361122
PartitionedCallП
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4                                    2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4                                    :& "
 
_user_specified_nameinputs
Бs
╟
G__inference_awsome_net_layer_call_and_return_conditional_losses_2736838

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
identityИв b_norm_1/StatefulPartitionedCallв b_norm_2/StatefulPartitionedCallв b_norm_3/StatefulPartitionedCallв b_norm_4/StatefulPartitionedCallвconv_1/StatefulPartitionedCallвconv_2/StatefulPartitionedCallвconv_3/StatefulPartitionedCallвconv_4/StatefulPartitionedCallв"dropout_28/StatefulPartitionedCallв"dropout_29/StatefulPartitionedCallв1input_batch_normalization/StatefulPartitionedCallв"input_conv/StatefulPartitionedCallв1output_class_distribution/StatefulPartitionedCall┬
"input_conv/StatefulPartitionedCallStatefulPartitionedCallinputs)input_conv_statefulpartitionedcall_args_1)input_conv_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*1
_output_shapes
:         АА *-
config_proto

GPU

CPU2*0J 8*P
fKRI
G__inference_input_conv_layer_call_and_return_conditional_losses_27353102$
"input_conv/StatefulPartitionedCallи
1input_batch_normalization/StatefulPartitionedCallStatefulPartitionedCall+input_conv/StatefulPartitionedCall:output:08input_batch_normalization_statefulpartitionedcall_args_18input_batch_normalization_statefulpartitionedcall_args_28input_batch_normalization_statefulpartitionedcall_args_38input_batch_normalization_statefulpartitionedcall_args_4*
Tin	
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*1
_output_shapes
:         АА *-
config_proto

GPU

CPU2*0J 8*_
fZRX
V__inference_input_batch_normalization_layer_call_and_return_conditional_losses_273615823
1input_batch_normalization/StatefulPartitionedCallЖ
input_RELU/PartitionedCallPartitionedCall:input_batch_normalization/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*1
_output_shapes
:         АА *-
config_proto

GPU

CPU2*0J 8*P
fKRI
G__inference_input_RELU_layer_call_and_return_conditional_losses_27362092
input_RELU/PartitionedCall 
 max_pooling2d_60/PartitionedCallPartitionedCall#input_RELU/PartitionedCall:output:0*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*/
_output_shapes
:         @@ *-
config_proto

GPU

CPU2*0J 8*V
fQRO
M__inference_max_pooling2d_60_layer_call_and_return_conditional_losses_27354562"
 max_pooling2d_60/PartitionedCallЛ
"dropout_28/StatefulPartitionedCallStatefulPartitionedCall)max_pooling2d_60/PartitionedCall:output:0*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*/
_output_shapes
:         @@ *-
config_proto

GPU

CPU2*0J 8*P
fKRI
G__inference_dropout_28_layer_call_and_return_conditional_losses_27362382$
"dropout_28/StatefulPartitionedCall╤
conv_1/StatefulPartitionedCallStatefulPartitionedCall+dropout_28/StatefulPartitionedCall:output:0%conv_1_statefulpartitionedcall_args_1%conv_1_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*/
_output_shapes
:         @@ *-
config_proto

GPU

CPU2*0J 8*L
fGRE
C__inference_conv_1_layer_call_and_return_conditional_losses_27354742 
conv_1/StatefulPartitionedCallл
 b_norm_1/StatefulPartitionedCallStatefulPartitionedCall'conv_1/StatefulPartitionedCall:output:0'b_norm_1_statefulpartitionedcall_args_1'b_norm_1_statefulpartitionedcall_args_2'b_norm_1_statefulpartitionedcall_args_3'b_norm_1_statefulpartitionedcall_args_4*
Tin	
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*/
_output_shapes
:         @@ *-
config_proto

GPU

CPU2*0J 8*N
fIRG
E__inference_b_norm_1_layer_call_and_return_conditional_losses_27362922"
 b_norm_1/StatefulPartitionedCallч
RELU_1/PartitionedCallPartitionedCall)b_norm_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*/
_output_shapes
:         @@ *-
config_proto

GPU

CPU2*0J 8*L
fGRE
C__inference_RELU_1_layer_call_and_return_conditional_losses_27363432
RELU_1/PartitionedCall√
 max_pooling2d_56/PartitionedCallPartitionedCallRELU_1/PartitionedCall:output:0*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*/
_output_shapes
:            *-
config_proto

GPU

CPU2*0J 8*V
fQRO
M__inference_max_pooling2d_56_layer_call_and_return_conditional_losses_27356202"
 max_pooling2d_56/PartitionedCall╧
conv_2/StatefulPartitionedCallStatefulPartitionedCall)max_pooling2d_56/PartitionedCall:output:0%conv_2_statefulpartitionedcall_args_1%conv_2_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*/
_output_shapes
:           @*-
config_proto

GPU

CPU2*0J 8*L
fGRE
C__inference_conv_2_layer_call_and_return_conditional_losses_27356382 
conv_2/StatefulPartitionedCallл
 b_norm_2/StatefulPartitionedCallStatefulPartitionedCall'conv_2/StatefulPartitionedCall:output:0'b_norm_2_statefulpartitionedcall_args_1'b_norm_2_statefulpartitionedcall_args_2'b_norm_2_statefulpartitionedcall_args_3'b_norm_2_statefulpartitionedcall_args_4*
Tin	
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*/
_output_shapes
:           @*-
config_proto

GPU

CPU2*0J 8*N
fIRG
E__inference_b_norm_2_layer_call_and_return_conditional_losses_27363882"
 b_norm_2/StatefulPartitionedCallч
RELU_2/PartitionedCallPartitionedCall)b_norm_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*/
_output_shapes
:           @*-
config_proto

GPU

CPU2*0J 8*L
fGRE
C__inference_RELU_2_layer_call_and_return_conditional_losses_27364392
RELU_2/PartitionedCall√
 max_pooling2d_57/PartitionedCallPartitionedCallRELU_2/PartitionedCall:output:0*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*/
_output_shapes
:         @*-
config_proto

GPU

CPU2*0J 8*V
fQRO
M__inference_max_pooling2d_57_layer_call_and_return_conditional_losses_27357842"
 max_pooling2d_57/PartitionedCall╨
conv_3/StatefulPartitionedCallStatefulPartitionedCall)max_pooling2d_57/PartitionedCall:output:0%conv_3_statefulpartitionedcall_args_1%conv_3_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*0
_output_shapes
:         А*-
config_proto

GPU

CPU2*0J 8*L
fGRE
C__inference_conv_3_layer_call_and_return_conditional_losses_27358022 
conv_3/StatefulPartitionedCallм
 b_norm_3/StatefulPartitionedCallStatefulPartitionedCall'conv_3/StatefulPartitionedCall:output:0'b_norm_3_statefulpartitionedcall_args_1'b_norm_3_statefulpartitionedcall_args_2'b_norm_3_statefulpartitionedcall_args_3'b_norm_3_statefulpartitionedcall_args_4*
Tin	
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*0
_output_shapes
:         А*-
config_proto

GPU

CPU2*0J 8*N
fIRG
E__inference_b_norm_3_layer_call_and_return_conditional_losses_27364842"
 b_norm_3/StatefulPartitionedCallш
RELU_3/PartitionedCallPartitionedCall)b_norm_3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*0
_output_shapes
:         А*-
config_proto

GPU

CPU2*0J 8*L
fGRE
C__inference_RELU_3_layer_call_and_return_conditional_losses_27365352
RELU_3/PartitionedCall№
 max_pooling2d_58/PartitionedCallPartitionedCallRELU_3/PartitionedCall:output:0*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*0
_output_shapes
:         А*-
config_proto

GPU

CPU2*0J 8*V
fQRO
M__inference_max_pooling2d_58_layer_call_and_return_conditional_losses_27359482"
 max_pooling2d_58/PartitionedCall╨
conv_4/StatefulPartitionedCallStatefulPartitionedCall)max_pooling2d_58/PartitionedCall:output:0%conv_4_statefulpartitionedcall_args_1%conv_4_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*0
_output_shapes
:         А*-
config_proto

GPU

CPU2*0J 8*L
fGRE
C__inference_conv_4_layer_call_and_return_conditional_losses_27359662 
conv_4/StatefulPartitionedCallм
 b_norm_4/StatefulPartitionedCallStatefulPartitionedCall'conv_4/StatefulPartitionedCall:output:0'b_norm_4_statefulpartitionedcall_args_1'b_norm_4_statefulpartitionedcall_args_2'b_norm_4_statefulpartitionedcall_args_3'b_norm_4_statefulpartitionedcall_args_4*
Tin	
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*0
_output_shapes
:         А*-
config_proto

GPU

CPU2*0J 8*N
fIRG
E__inference_b_norm_4_layer_call_and_return_conditional_losses_27365802"
 b_norm_4/StatefulPartitionedCallш
RELU_4/PartitionedCallPartitionedCall)b_norm_4/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*0
_output_shapes
:         А*-
config_proto

GPU

CPU2*0J 8*L
fGRE
C__inference_RELU_4_layer_call_and_return_conditional_losses_27366312
RELU_4/PartitionedCall№
 max_pooling2d_59/PartitionedCallPartitionedCallRELU_4/PartitionedCall:output:0*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*0
_output_shapes
:         А*-
config_proto

GPU

CPU2*0J 8*V
fQRO
M__inference_max_pooling2d_59_layer_call_and_return_conditional_losses_27361122"
 max_pooling2d_59/PartitionedCallь
flatten_14/PartitionedCallPartitionedCall)max_pooling2d_59/PartitionedCall:output:0*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*(
_output_shapes
:         А*-
config_proto

GPU

CPU2*0J 8*P
fKRI
G__inference_flatten_14_layer_call_and_return_conditional_losses_27366462
flatten_14/PartitionedCallг
"dropout_29/StatefulPartitionedCallStatefulPartitionedCall#flatten_14/PartitionedCall:output:0#^dropout_28/StatefulPartitionedCall*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*(
_output_shapes
:         А*-
config_proto

GPU

CPU2*0J 8*P
fKRI
G__inference_dropout_29_layer_call_and_return_conditional_losses_27366742$
"dropout_29/StatefulPartitionedCallй
1output_class_distribution/StatefulPartitionedCallStatefulPartitionedCall+dropout_29/StatefulPartitionedCall:output:08output_class_distribution_statefulpartitionedcall_args_18output_class_distribution_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*(
_output_shapes
:         ╚*-
config_proto

GPU

CPU2*0J 8*_
fZRX
V__inference_output_class_distribution_layer_call_and_return_conditional_losses_273670223
1output_class_distribution/StatefulPartitionedCallЎ
IdentityIdentity:output_class_distribution/StatefulPartitionedCall:output:0!^b_norm_1/StatefulPartitionedCall!^b_norm_2/StatefulPartitionedCall!^b_norm_3/StatefulPartitionedCall!^b_norm_4/StatefulPartitionedCall^conv_1/StatefulPartitionedCall^conv_2/StatefulPartitionedCall^conv_3/StatefulPartitionedCall^conv_4/StatefulPartitionedCall#^dropout_28/StatefulPartitionedCall#^dropout_29/StatefulPartitionedCall2^input_batch_normalization/StatefulPartitionedCall#^input_conv/StatefulPartitionedCall2^output_class_distribution/StatefulPartitionedCall*
T0*(
_output_shapes
:         ╚2

Identity"
identityIdentity:output:0*▓
_input_shapesа
Э:         АА::::::::::::::::::::::::::::::::2D
 b_norm_1/StatefulPartitionedCall b_norm_1/StatefulPartitionedCall2D
 b_norm_2/StatefulPartitionedCall b_norm_2/StatefulPartitionedCall2D
 b_norm_3/StatefulPartitionedCall b_norm_3/StatefulPartitionedCall2D
 b_norm_4/StatefulPartitionedCall b_norm_4/StatefulPartitionedCall2@
conv_1/StatefulPartitionedCallconv_1/StatefulPartitionedCall2@
conv_2/StatefulPartitionedCallconv_2/StatefulPartitionedCall2@
conv_3/StatefulPartitionedCallconv_3/StatefulPartitionedCall2@
conv_4/StatefulPartitionedCallconv_4/StatefulPartitionedCall2H
"dropout_28/StatefulPartitionedCall"dropout_28/StatefulPartitionedCall2H
"dropout_29/StatefulPartitionedCall"dropout_29/StatefulPartitionedCall2f
1input_batch_normalization/StatefulPartitionedCall1input_batch_normalization/StatefulPartitionedCall2H
"input_conv/StatefulPartitionedCall"input_conv/StatefulPartitionedCall2f
1output_class_distribution/StatefulPartitionedCall1output_class_distribution/StatefulPartitionedCall:& "
 
_user_specified_nameinputs
а
_
C__inference_RELU_2_layer_call_and_return_conditional_losses_2736439

inputs
identityV
ReluReluinputs*
T0*/
_output_shapes
:           @2
Relun
IdentityIdentityRelu:activations:0*
T0*/
_output_shapes
:           @2

Identity"
identityIdentity:output:0*.
_input_shapes
:           @:& "
 
_user_specified_nameinputs
ё

▄
C__inference_conv_2_layer_call_and_return_conditional_losses_2735638

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identityИвBiasAdd/ReadVariableOpвConv2D/ReadVariableOpo
dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      2
dilation_rateХ
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: @*
dtype02
Conv2D/ReadVariableOp╡
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+                           @*
paddingSAME*
strides
2
Conv2DМ
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOpЪ
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+                           @2	
BiasAddп
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*A
_output_shapes/
-:+                           @2

Identity"
identityIdentity:output:0*H
_input_shapes7
5:+                            ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:& "
 
_user_specified_nameinputs
а
_
C__inference_RELU_1_layer_call_and_return_conditional_losses_2737898

inputs
identityV
ReluReluinputs*
T0*/
_output_shapes
:         @@ 2
Relun
IdentityIdentityRelu:activations:0*
T0*/
_output_shapes
:         @@ 2

Identity"
identityIdentity:output:0*.
_input_shapes
:         @@ :& "
 
_user_specified_nameinputs
┤╧
Ш
G__inference_awsome_net_layer_call_and_return_conditional_losses_2737310

inputs-
)input_conv_conv2d_readvariableop_resource.
*input_conv_biasadd_readvariableop_resource5
1input_batch_normalization_readvariableop_resource7
3input_batch_normalization_readvariableop_1_resource5
1input_batch_normalization_assignmovingavg_27371017
3input_batch_normalization_assignmovingavg_1_2737108)
%conv_1_conv2d_readvariableop_resource*
&conv_1_biasadd_readvariableop_resource$
 b_norm_1_readvariableop_resource&
"b_norm_1_readvariableop_1_resource$
 b_norm_1_assignmovingavg_2737155&
"b_norm_1_assignmovingavg_1_2737162)
%conv_2_conv2d_readvariableop_resource*
&conv_2_biasadd_readvariableop_resource$
 b_norm_2_readvariableop_resource&
"b_norm_2_readvariableop_1_resource$
 b_norm_2_assignmovingavg_2737193&
"b_norm_2_assignmovingavg_1_2737200)
%conv_3_conv2d_readvariableop_resource*
&conv_3_biasadd_readvariableop_resource$
 b_norm_3_readvariableop_resource&
"b_norm_3_readvariableop_1_resource$
 b_norm_3_assignmovingavg_2737231&
"b_norm_3_assignmovingavg_1_2737238)
%conv_4_conv2d_readvariableop_resource*
&conv_4_biasadd_readvariableop_resource$
 b_norm_4_readvariableop_resource&
"b_norm_4_readvariableop_1_resource$
 b_norm_4_assignmovingavg_2737269&
"b_norm_4_assignmovingavg_1_2737276<
8output_class_distribution_matmul_readvariableop_resource=
9output_class_distribution_biasadd_readvariableop_resource
identityИв,b_norm_1/AssignMovingAvg/AssignSubVariableOpв'b_norm_1/AssignMovingAvg/ReadVariableOpв.b_norm_1/AssignMovingAvg_1/AssignSubVariableOpв)b_norm_1/AssignMovingAvg_1/ReadVariableOpвb_norm_1/ReadVariableOpвb_norm_1/ReadVariableOp_1в,b_norm_2/AssignMovingAvg/AssignSubVariableOpв'b_norm_2/AssignMovingAvg/ReadVariableOpв.b_norm_2/AssignMovingAvg_1/AssignSubVariableOpв)b_norm_2/AssignMovingAvg_1/ReadVariableOpвb_norm_2/ReadVariableOpвb_norm_2/ReadVariableOp_1в,b_norm_3/AssignMovingAvg/AssignSubVariableOpв'b_norm_3/AssignMovingAvg/ReadVariableOpв.b_norm_3/AssignMovingAvg_1/AssignSubVariableOpв)b_norm_3/AssignMovingAvg_1/ReadVariableOpвb_norm_3/ReadVariableOpвb_norm_3/ReadVariableOp_1в,b_norm_4/AssignMovingAvg/AssignSubVariableOpв'b_norm_4/AssignMovingAvg/ReadVariableOpв.b_norm_4/AssignMovingAvg_1/AssignSubVariableOpв)b_norm_4/AssignMovingAvg_1/ReadVariableOpвb_norm_4/ReadVariableOpвb_norm_4/ReadVariableOp_1вconv_1/BiasAdd/ReadVariableOpвconv_1/Conv2D/ReadVariableOpвconv_2/BiasAdd/ReadVariableOpвconv_2/Conv2D/ReadVariableOpвconv_3/BiasAdd/ReadVariableOpвconv_3/Conv2D/ReadVariableOpвconv_4/BiasAdd/ReadVariableOpвconv_4/Conv2D/ReadVariableOpв=input_batch_normalization/AssignMovingAvg/AssignSubVariableOpв8input_batch_normalization/AssignMovingAvg/ReadVariableOpв?input_batch_normalization/AssignMovingAvg_1/AssignSubVariableOpв:input_batch_normalization/AssignMovingAvg_1/ReadVariableOpв(input_batch_normalization/ReadVariableOpв*input_batch_normalization/ReadVariableOp_1в!input_conv/BiasAdd/ReadVariableOpв input_conv/Conv2D/ReadVariableOpв0output_class_distribution/BiasAdd/ReadVariableOpв/output_class_distribution/MatMul/ReadVariableOp╢
 input_conv/Conv2D/ReadVariableOpReadVariableOp)input_conv_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02"
 input_conv/Conv2D/ReadVariableOp╞
input_conv/Conv2DConv2Dinputs(input_conv/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:         АА *
paddingSAME*
strides
2
input_conv/Conv2Dн
!input_conv/BiasAdd/ReadVariableOpReadVariableOp*input_conv_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02#
!input_conv/BiasAdd/ReadVariableOp╢
input_conv/BiasAddBiasAddinput_conv/Conv2D:output:0)input_conv/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:         АА 2
input_conv/BiasAddТ
&input_batch_normalization/LogicalAnd/xConst*
_output_shapes
: *
dtype0
*
value	B
 Z2(
&input_batch_normalization/LogicalAnd/xТ
&input_batch_normalization/LogicalAnd/yConst*
_output_shapes
: *
dtype0
*
value	B
 Z2(
&input_batch_normalization/LogicalAnd/y╘
$input_batch_normalization/LogicalAnd
LogicalAnd/input_batch_normalization/LogicalAnd/x:output:0/input_batch_normalization/LogicalAnd/y:output:0*
_output_shapes
: 2&
$input_batch_normalization/LogicalAnd┬
(input_batch_normalization/ReadVariableOpReadVariableOp1input_batch_normalization_readvariableop_resource*
_output_shapes
: *
dtype02*
(input_batch_normalization/ReadVariableOp╚
*input_batch_normalization/ReadVariableOp_1ReadVariableOp3input_batch_normalization_readvariableop_1_resource*
_output_shapes
: *
dtype02,
*input_batch_normalization/ReadVariableOp_1Е
input_batch_normalization/ConstConst*
_output_shapes
: *
dtype0*
valueB 2!
input_batch_normalization/ConstЙ
!input_batch_normalization/Const_1Const*
_output_shapes
: *
dtype0*
valueB 2#
!input_batch_normalization/Const_1╕
*input_batch_normalization/FusedBatchNormV3FusedBatchNormV3input_conv/BiasAdd:output:00input_batch_normalization/ReadVariableOp:value:02input_batch_normalization/ReadVariableOp_1:value:0(input_batch_normalization/Const:output:0*input_batch_normalization/Const_1:output:0*
T0*
U0*M
_output_shapes;
9:         АА : : : : :*
epsilon%oГ:2,
*input_batch_normalization/FusedBatchNormV3Л
!input_batch_normalization/Const_2Const*
_output_shapes
: *
dtype0*
valueB
 *дp}?2#
!input_batch_normalization/Const_2э
/input_batch_normalization/AssignMovingAvg/sub/xConst*D
_class:
86loc:@input_batch_normalization/AssignMovingAvg/2737101*
_output_shapes
: *
dtype0*
valueB
 *  А?21
/input_batch_normalization/AssignMovingAvg/sub/x▓
-input_batch_normalization/AssignMovingAvg/subSub8input_batch_normalization/AssignMovingAvg/sub/x:output:0*input_batch_normalization/Const_2:output:0*
T0*D
_class:
86loc:@input_batch_normalization/AssignMovingAvg/2737101*
_output_shapes
: 2/
-input_batch_normalization/AssignMovingAvg/subт
8input_batch_normalization/AssignMovingAvg/ReadVariableOpReadVariableOp1input_batch_normalization_assignmovingavg_2737101*
_output_shapes
: *
dtype02:
8input_batch_normalization/AssignMovingAvg/ReadVariableOp╧
/input_batch_normalization/AssignMovingAvg/sub_1Sub@input_batch_normalization/AssignMovingAvg/ReadVariableOp:value:07input_batch_normalization/FusedBatchNormV3:batch_mean:0*
T0*D
_class:
86loc:@input_batch_normalization/AssignMovingAvg/2737101*
_output_shapes
: 21
/input_batch_normalization/AssignMovingAvg/sub_1╕
-input_batch_normalization/AssignMovingAvg/mulMul3input_batch_normalization/AssignMovingAvg/sub_1:z:01input_batch_normalization/AssignMovingAvg/sub:z:0*
T0*D
_class:
86loc:@input_batch_normalization/AssignMovingAvg/2737101*
_output_shapes
: 2/
-input_batch_normalization/AssignMovingAvg/mulЯ
=input_batch_normalization/AssignMovingAvg/AssignSubVariableOpAssignSubVariableOp1input_batch_normalization_assignmovingavg_27371011input_batch_normalization/AssignMovingAvg/mul:z:09^input_batch_normalization/AssignMovingAvg/ReadVariableOp*D
_class:
86loc:@input_batch_normalization/AssignMovingAvg/2737101*
_output_shapes
 *
dtype02?
=input_batch_normalization/AssignMovingAvg/AssignSubVariableOpє
1input_batch_normalization/AssignMovingAvg_1/sub/xConst*F
_class<
:8loc:@input_batch_normalization/AssignMovingAvg_1/2737108*
_output_shapes
: *
dtype0*
valueB
 *  А?23
1input_batch_normalization/AssignMovingAvg_1/sub/x║
/input_batch_normalization/AssignMovingAvg_1/subSub:input_batch_normalization/AssignMovingAvg_1/sub/x:output:0*input_batch_normalization/Const_2:output:0*
T0*F
_class<
:8loc:@input_batch_normalization/AssignMovingAvg_1/2737108*
_output_shapes
: 21
/input_batch_normalization/AssignMovingAvg_1/subш
:input_batch_normalization/AssignMovingAvg_1/ReadVariableOpReadVariableOp3input_batch_normalization_assignmovingavg_1_2737108*
_output_shapes
: *
dtype02<
:input_batch_normalization/AssignMovingAvg_1/ReadVariableOp█
1input_batch_normalization/AssignMovingAvg_1/sub_1SubBinput_batch_normalization/AssignMovingAvg_1/ReadVariableOp:value:0;input_batch_normalization/FusedBatchNormV3:batch_variance:0*
T0*F
_class<
:8loc:@input_batch_normalization/AssignMovingAvg_1/2737108*
_output_shapes
: 23
1input_batch_normalization/AssignMovingAvg_1/sub_1┬
/input_batch_normalization/AssignMovingAvg_1/mulMul5input_batch_normalization/AssignMovingAvg_1/sub_1:z:03input_batch_normalization/AssignMovingAvg_1/sub:z:0*
T0*F
_class<
:8loc:@input_batch_normalization/AssignMovingAvg_1/2737108*
_output_shapes
: 21
/input_batch_normalization/AssignMovingAvg_1/mulл
?input_batch_normalization/AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOp3input_batch_normalization_assignmovingavg_1_27371083input_batch_normalization/AssignMovingAvg_1/mul:z:0;^input_batch_normalization/AssignMovingAvg_1/ReadVariableOp*F
_class<
:8loc:@input_batch_normalization/AssignMovingAvg_1/2737108*
_output_shapes
 *
dtype02A
?input_batch_normalization/AssignMovingAvg_1/AssignSubVariableOpЦ
input_RELU/ReluRelu.input_batch_normalization/FusedBatchNormV3:y:0*
T0*1
_output_shapes
:         АА 2
input_RELU/Relu╦
max_pooling2d_60/MaxPoolMaxPoolinput_RELU/Relu:activations:0*/
_output_shapes
:         @@ *
ksize
*
paddingVALID*
strides
2
max_pooling2d_60/MaxPoolw
dropout_28/dropout/rateConst*
_output_shapes
: *
dtype0*
valueB
 *  А>2
dropout_28/dropout/rateЕ
dropout_28/dropout/ShapeShape!max_pooling2d_60/MaxPool:output:0*
T0*
_output_shapes
:2
dropout_28/dropout/ShapeУ
%dropout_28/dropout/random_uniform/minConst*
_output_shapes
: *
dtype0*
valueB
 *    2'
%dropout_28/dropout/random_uniform/minУ
%dropout_28/dropout/random_uniform/maxConst*
_output_shapes
: *
dtype0*
valueB
 *  А?2'
%dropout_28/dropout/random_uniform/max▌
/dropout_28/dropout/random_uniform/RandomUniformRandomUniform!dropout_28/dropout/Shape:output:0*
T0*/
_output_shapes
:         @@ *
dtype021
/dropout_28/dropout/random_uniform/RandomUniform╓
%dropout_28/dropout/random_uniform/subSub.dropout_28/dropout/random_uniform/max:output:0.dropout_28/dropout/random_uniform/min:output:0*
T0*
_output_shapes
: 2'
%dropout_28/dropout/random_uniform/subЇ
%dropout_28/dropout/random_uniform/mulMul8dropout_28/dropout/random_uniform/RandomUniform:output:0)dropout_28/dropout/random_uniform/sub:z:0*
T0*/
_output_shapes
:         @@ 2'
%dropout_28/dropout/random_uniform/mulт
!dropout_28/dropout/random_uniformAdd)dropout_28/dropout/random_uniform/mul:z:0.dropout_28/dropout/random_uniform/min:output:0*
T0*/
_output_shapes
:         @@ 2#
!dropout_28/dropout/random_uniformy
dropout_28/dropout/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  А?2
dropout_28/dropout/sub/xЭ
dropout_28/dropout/subSub!dropout_28/dropout/sub/x:output:0 dropout_28/dropout/rate:output:0*
T0*
_output_shapes
: 2
dropout_28/dropout/subБ
dropout_28/dropout/truediv/xConst*
_output_shapes
: *
dtype0*
valueB
 *  А?2
dropout_28/dropout/truediv/xз
dropout_28/dropout/truedivRealDiv%dropout_28/dropout/truediv/x:output:0dropout_28/dropout/sub:z:0*
T0*
_output_shapes
: 2
dropout_28/dropout/truediv╒
dropout_28/dropout/GreaterEqualGreaterEqual%dropout_28/dropout/random_uniform:z:0 dropout_28/dropout/rate:output:0*
T0*/
_output_shapes
:         @@ 2!
dropout_28/dropout/GreaterEqual┤
dropout_28/dropout/mulMul!max_pooling2d_60/MaxPool:output:0dropout_28/dropout/truediv:z:0*
T0*/
_output_shapes
:         @@ 2
dropout_28/dropout/mulи
dropout_28/dropout/CastCast#dropout_28/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:         @@ 2
dropout_28/dropout/Castо
dropout_28/dropout/mul_1Muldropout_28/dropout/mul:z:0dropout_28/dropout/Cast:y:0*
T0*/
_output_shapes
:         @@ 2
dropout_28/dropout/mul_1к
conv_1/Conv2D/ReadVariableOpReadVariableOp%conv_1_conv2d_readvariableop_resource*&
_output_shapes
:  *
dtype02
conv_1/Conv2D/ReadVariableOp╬
conv_1/Conv2DConv2Ddropout_28/dropout/mul_1:z:0$conv_1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         @@ *
paddingSAME*
strides
2
conv_1/Conv2Dб
conv_1/BiasAdd/ReadVariableOpReadVariableOp&conv_1_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02
conv_1/BiasAdd/ReadVariableOpд
conv_1/BiasAddBiasAddconv_1/Conv2D:output:0%conv_1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         @@ 2
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
b_norm_1/LogicalAnd/yР
b_norm_1/LogicalAnd
LogicalAndb_norm_1/LogicalAnd/x:output:0b_norm_1/LogicalAnd/y:output:0*
_output_shapes
: 2
b_norm_1/LogicalAndП
b_norm_1/ReadVariableOpReadVariableOp b_norm_1_readvariableop_resource*
_output_shapes
: *
dtype02
b_norm_1/ReadVariableOpХ
b_norm_1/ReadVariableOp_1ReadVariableOp"b_norm_1_readvariableop_1_resource*
_output_shapes
: *
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
b_norm_1/Const_1╠
b_norm_1/FusedBatchNormV3FusedBatchNormV3conv_1/BiasAdd:output:0b_norm_1/ReadVariableOp:value:0!b_norm_1/ReadVariableOp_1:value:0b_norm_1/Const:output:0b_norm_1/Const_1:output:0*
T0*
U0*K
_output_shapes9
7:         @@ : : : : :*
epsilon%oГ:2
b_norm_1/FusedBatchNormV3i
b_norm_1/Const_2Const*
_output_shapes
: *
dtype0*
valueB
 *дp}?2
b_norm_1/Const_2║
b_norm_1/AssignMovingAvg/sub/xConst*3
_class)
'%loc:@b_norm_1/AssignMovingAvg/2737155*
_output_shapes
: *
dtype0*
valueB
 *  А?2 
b_norm_1/AssignMovingAvg/sub/x▌
b_norm_1/AssignMovingAvg/subSub'b_norm_1/AssignMovingAvg/sub/x:output:0b_norm_1/Const_2:output:0*
T0*3
_class)
'%loc:@b_norm_1/AssignMovingAvg/2737155*
_output_shapes
: 2
b_norm_1/AssignMovingAvg/subп
'b_norm_1/AssignMovingAvg/ReadVariableOpReadVariableOp b_norm_1_assignmovingavg_2737155*
_output_shapes
: *
dtype02)
'b_norm_1/AssignMovingAvg/ReadVariableOp·
b_norm_1/AssignMovingAvg/sub_1Sub/b_norm_1/AssignMovingAvg/ReadVariableOp:value:0&b_norm_1/FusedBatchNormV3:batch_mean:0*
T0*3
_class)
'%loc:@b_norm_1/AssignMovingAvg/2737155*
_output_shapes
: 2 
b_norm_1/AssignMovingAvg/sub_1у
b_norm_1/AssignMovingAvg/mulMul"b_norm_1/AssignMovingAvg/sub_1:z:0 b_norm_1/AssignMovingAvg/sub:z:0*
T0*3
_class)
'%loc:@b_norm_1/AssignMovingAvg/2737155*
_output_shapes
: 2
b_norm_1/AssignMovingAvg/mul╣
,b_norm_1/AssignMovingAvg/AssignSubVariableOpAssignSubVariableOp b_norm_1_assignmovingavg_2737155 b_norm_1/AssignMovingAvg/mul:z:0(^b_norm_1/AssignMovingAvg/ReadVariableOp*3
_class)
'%loc:@b_norm_1/AssignMovingAvg/2737155*
_output_shapes
 *
dtype02.
,b_norm_1/AssignMovingAvg/AssignSubVariableOp└
 b_norm_1/AssignMovingAvg_1/sub/xConst*5
_class+
)'loc:@b_norm_1/AssignMovingAvg_1/2737162*
_output_shapes
: *
dtype0*
valueB
 *  А?2"
 b_norm_1/AssignMovingAvg_1/sub/xх
b_norm_1/AssignMovingAvg_1/subSub)b_norm_1/AssignMovingAvg_1/sub/x:output:0b_norm_1/Const_2:output:0*
T0*5
_class+
)'loc:@b_norm_1/AssignMovingAvg_1/2737162*
_output_shapes
: 2 
b_norm_1/AssignMovingAvg_1/sub╡
)b_norm_1/AssignMovingAvg_1/ReadVariableOpReadVariableOp"b_norm_1_assignmovingavg_1_2737162*
_output_shapes
: *
dtype02+
)b_norm_1/AssignMovingAvg_1/ReadVariableOpЖ
 b_norm_1/AssignMovingAvg_1/sub_1Sub1b_norm_1/AssignMovingAvg_1/ReadVariableOp:value:0*b_norm_1/FusedBatchNormV3:batch_variance:0*
T0*5
_class+
)'loc:@b_norm_1/AssignMovingAvg_1/2737162*
_output_shapes
: 2"
 b_norm_1/AssignMovingAvg_1/sub_1э
b_norm_1/AssignMovingAvg_1/mulMul$b_norm_1/AssignMovingAvg_1/sub_1:z:0"b_norm_1/AssignMovingAvg_1/sub:z:0*
T0*5
_class+
)'loc:@b_norm_1/AssignMovingAvg_1/2737162*
_output_shapes
: 2 
b_norm_1/AssignMovingAvg_1/mul┼
.b_norm_1/AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOp"b_norm_1_assignmovingavg_1_2737162"b_norm_1/AssignMovingAvg_1/mul:z:0*^b_norm_1/AssignMovingAvg_1/ReadVariableOp*5
_class+
)'loc:@b_norm_1/AssignMovingAvg_1/2737162*
_output_shapes
 *
dtype020
.b_norm_1/AssignMovingAvg_1/AssignSubVariableOp{
RELU_1/ReluRelub_norm_1/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:         @@ 2
RELU_1/Relu╟
max_pooling2d_56/MaxPoolMaxPoolRELU_1/Relu:activations:0*/
_output_shapes
:            *
ksize
*
paddingVALID*
strides
2
max_pooling2d_56/MaxPoolк
conv_2/Conv2D/ReadVariableOpReadVariableOp%conv_2_conv2d_readvariableop_resource*&
_output_shapes
: @*
dtype02
conv_2/Conv2D/ReadVariableOp╙
conv_2/Conv2DConv2D!max_pooling2d_56/MaxPool:output:0$conv_2/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:           @*
paddingSAME*
strides
2
conv_2/Conv2Dб
conv_2/BiasAdd/ReadVariableOpReadVariableOp&conv_2_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
conv_2/BiasAdd/ReadVariableOpд
conv_2/BiasAddBiasAddconv_2/Conv2D:output:0%conv_2/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:           @2
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
b_norm_2/LogicalAnd/yР
b_norm_2/LogicalAnd
LogicalAndb_norm_2/LogicalAnd/x:output:0b_norm_2/LogicalAnd/y:output:0*
_output_shapes
: 2
b_norm_2/LogicalAndП
b_norm_2/ReadVariableOpReadVariableOp b_norm_2_readvariableop_resource*
_output_shapes
:@*
dtype02
b_norm_2/ReadVariableOpХ
b_norm_2/ReadVariableOp_1ReadVariableOp"b_norm_2_readvariableop_1_resource*
_output_shapes
:@*
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
b_norm_2/Const_1╠
b_norm_2/FusedBatchNormV3FusedBatchNormV3conv_2/BiasAdd:output:0b_norm_2/ReadVariableOp:value:0!b_norm_2/ReadVariableOp_1:value:0b_norm_2/Const:output:0b_norm_2/Const_1:output:0*
T0*
U0*K
_output_shapes9
7:           @:@:@:@:@:*
epsilon%oГ:2
b_norm_2/FusedBatchNormV3i
b_norm_2/Const_2Const*
_output_shapes
: *
dtype0*
valueB
 *дp}?2
b_norm_2/Const_2║
b_norm_2/AssignMovingAvg/sub/xConst*3
_class)
'%loc:@b_norm_2/AssignMovingAvg/2737193*
_output_shapes
: *
dtype0*
valueB
 *  А?2 
b_norm_2/AssignMovingAvg/sub/x▌
b_norm_2/AssignMovingAvg/subSub'b_norm_2/AssignMovingAvg/sub/x:output:0b_norm_2/Const_2:output:0*
T0*3
_class)
'%loc:@b_norm_2/AssignMovingAvg/2737193*
_output_shapes
: 2
b_norm_2/AssignMovingAvg/subп
'b_norm_2/AssignMovingAvg/ReadVariableOpReadVariableOp b_norm_2_assignmovingavg_2737193*
_output_shapes
:@*
dtype02)
'b_norm_2/AssignMovingAvg/ReadVariableOp·
b_norm_2/AssignMovingAvg/sub_1Sub/b_norm_2/AssignMovingAvg/ReadVariableOp:value:0&b_norm_2/FusedBatchNormV3:batch_mean:0*
T0*3
_class)
'%loc:@b_norm_2/AssignMovingAvg/2737193*
_output_shapes
:@2 
b_norm_2/AssignMovingAvg/sub_1у
b_norm_2/AssignMovingAvg/mulMul"b_norm_2/AssignMovingAvg/sub_1:z:0 b_norm_2/AssignMovingAvg/sub:z:0*
T0*3
_class)
'%loc:@b_norm_2/AssignMovingAvg/2737193*
_output_shapes
:@2
b_norm_2/AssignMovingAvg/mul╣
,b_norm_2/AssignMovingAvg/AssignSubVariableOpAssignSubVariableOp b_norm_2_assignmovingavg_2737193 b_norm_2/AssignMovingAvg/mul:z:0(^b_norm_2/AssignMovingAvg/ReadVariableOp*3
_class)
'%loc:@b_norm_2/AssignMovingAvg/2737193*
_output_shapes
 *
dtype02.
,b_norm_2/AssignMovingAvg/AssignSubVariableOp└
 b_norm_2/AssignMovingAvg_1/sub/xConst*5
_class+
)'loc:@b_norm_2/AssignMovingAvg_1/2737200*
_output_shapes
: *
dtype0*
valueB
 *  А?2"
 b_norm_2/AssignMovingAvg_1/sub/xх
b_norm_2/AssignMovingAvg_1/subSub)b_norm_2/AssignMovingAvg_1/sub/x:output:0b_norm_2/Const_2:output:0*
T0*5
_class+
)'loc:@b_norm_2/AssignMovingAvg_1/2737200*
_output_shapes
: 2 
b_norm_2/AssignMovingAvg_1/sub╡
)b_norm_2/AssignMovingAvg_1/ReadVariableOpReadVariableOp"b_norm_2_assignmovingavg_1_2737200*
_output_shapes
:@*
dtype02+
)b_norm_2/AssignMovingAvg_1/ReadVariableOpЖ
 b_norm_2/AssignMovingAvg_1/sub_1Sub1b_norm_2/AssignMovingAvg_1/ReadVariableOp:value:0*b_norm_2/FusedBatchNormV3:batch_variance:0*
T0*5
_class+
)'loc:@b_norm_2/AssignMovingAvg_1/2737200*
_output_shapes
:@2"
 b_norm_2/AssignMovingAvg_1/sub_1э
b_norm_2/AssignMovingAvg_1/mulMul$b_norm_2/AssignMovingAvg_1/sub_1:z:0"b_norm_2/AssignMovingAvg_1/sub:z:0*
T0*5
_class+
)'loc:@b_norm_2/AssignMovingAvg_1/2737200*
_output_shapes
:@2 
b_norm_2/AssignMovingAvg_1/mul┼
.b_norm_2/AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOp"b_norm_2_assignmovingavg_1_2737200"b_norm_2/AssignMovingAvg_1/mul:z:0*^b_norm_2/AssignMovingAvg_1/ReadVariableOp*5
_class+
)'loc:@b_norm_2/AssignMovingAvg_1/2737200*
_output_shapes
 *
dtype020
.b_norm_2/AssignMovingAvg_1/AssignSubVariableOp{
RELU_2/ReluRelub_norm_2/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:           @2
RELU_2/Relu╟
max_pooling2d_57/MaxPoolMaxPoolRELU_2/Relu:activations:0*/
_output_shapes
:         @*
ksize
*
paddingVALID*
strides
2
max_pooling2d_57/MaxPoolл
conv_3/Conv2D/ReadVariableOpReadVariableOp%conv_3_conv2d_readvariableop_resource*'
_output_shapes
:@А*
dtype02
conv_3/Conv2D/ReadVariableOp╘
conv_3/Conv2DConv2D!max_pooling2d_57/MaxPool:output:0$conv_3/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:         А*
paddingSAME*
strides
2
conv_3/Conv2Dв
conv_3/BiasAdd/ReadVariableOpReadVariableOp&conv_3_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02
conv_3/BiasAdd/ReadVariableOpе
conv_3/BiasAddBiasAddconv_3/Conv2D:output:0%conv_3/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:         А2
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
b_norm_3/LogicalAnd/yР
b_norm_3/LogicalAnd
LogicalAndb_norm_3/LogicalAnd/x:output:0b_norm_3/LogicalAnd/y:output:0*
_output_shapes
: 2
b_norm_3/LogicalAndР
b_norm_3/ReadVariableOpReadVariableOp b_norm_3_readvariableop_resource*
_output_shapes	
:А*
dtype02
b_norm_3/ReadVariableOpЦ
b_norm_3/ReadVariableOp_1ReadVariableOp"b_norm_3_readvariableop_1_resource*
_output_shapes	
:А*
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
b_norm_3/Const_1╤
b_norm_3/FusedBatchNormV3FusedBatchNormV3conv_3/BiasAdd:output:0b_norm_3/ReadVariableOp:value:0!b_norm_3/ReadVariableOp_1:value:0b_norm_3/Const:output:0b_norm_3/Const_1:output:0*
T0*
U0*P
_output_shapes>
<:         А:А:А:А:А:*
epsilon%oГ:2
b_norm_3/FusedBatchNormV3i
b_norm_3/Const_2Const*
_output_shapes
: *
dtype0*
valueB
 *дp}?2
b_norm_3/Const_2║
b_norm_3/AssignMovingAvg/sub/xConst*3
_class)
'%loc:@b_norm_3/AssignMovingAvg/2737231*
_output_shapes
: *
dtype0*
valueB
 *  А?2 
b_norm_3/AssignMovingAvg/sub/x▌
b_norm_3/AssignMovingAvg/subSub'b_norm_3/AssignMovingAvg/sub/x:output:0b_norm_3/Const_2:output:0*
T0*3
_class)
'%loc:@b_norm_3/AssignMovingAvg/2737231*
_output_shapes
: 2
b_norm_3/AssignMovingAvg/sub░
'b_norm_3/AssignMovingAvg/ReadVariableOpReadVariableOp b_norm_3_assignmovingavg_2737231*
_output_shapes	
:А*
dtype02)
'b_norm_3/AssignMovingAvg/ReadVariableOp√
b_norm_3/AssignMovingAvg/sub_1Sub/b_norm_3/AssignMovingAvg/ReadVariableOp:value:0&b_norm_3/FusedBatchNormV3:batch_mean:0*
T0*3
_class)
'%loc:@b_norm_3/AssignMovingAvg/2737231*
_output_shapes	
:А2 
b_norm_3/AssignMovingAvg/sub_1ф
b_norm_3/AssignMovingAvg/mulMul"b_norm_3/AssignMovingAvg/sub_1:z:0 b_norm_3/AssignMovingAvg/sub:z:0*
T0*3
_class)
'%loc:@b_norm_3/AssignMovingAvg/2737231*
_output_shapes	
:А2
b_norm_3/AssignMovingAvg/mul╣
,b_norm_3/AssignMovingAvg/AssignSubVariableOpAssignSubVariableOp b_norm_3_assignmovingavg_2737231 b_norm_3/AssignMovingAvg/mul:z:0(^b_norm_3/AssignMovingAvg/ReadVariableOp*3
_class)
'%loc:@b_norm_3/AssignMovingAvg/2737231*
_output_shapes
 *
dtype02.
,b_norm_3/AssignMovingAvg/AssignSubVariableOp└
 b_norm_3/AssignMovingAvg_1/sub/xConst*5
_class+
)'loc:@b_norm_3/AssignMovingAvg_1/2737238*
_output_shapes
: *
dtype0*
valueB
 *  А?2"
 b_norm_3/AssignMovingAvg_1/sub/xх
b_norm_3/AssignMovingAvg_1/subSub)b_norm_3/AssignMovingAvg_1/sub/x:output:0b_norm_3/Const_2:output:0*
T0*5
_class+
)'loc:@b_norm_3/AssignMovingAvg_1/2737238*
_output_shapes
: 2 
b_norm_3/AssignMovingAvg_1/sub╢
)b_norm_3/AssignMovingAvg_1/ReadVariableOpReadVariableOp"b_norm_3_assignmovingavg_1_2737238*
_output_shapes	
:А*
dtype02+
)b_norm_3/AssignMovingAvg_1/ReadVariableOpЗ
 b_norm_3/AssignMovingAvg_1/sub_1Sub1b_norm_3/AssignMovingAvg_1/ReadVariableOp:value:0*b_norm_3/FusedBatchNormV3:batch_variance:0*
T0*5
_class+
)'loc:@b_norm_3/AssignMovingAvg_1/2737238*
_output_shapes	
:А2"
 b_norm_3/AssignMovingAvg_1/sub_1ю
b_norm_3/AssignMovingAvg_1/mulMul$b_norm_3/AssignMovingAvg_1/sub_1:z:0"b_norm_3/AssignMovingAvg_1/sub:z:0*
T0*5
_class+
)'loc:@b_norm_3/AssignMovingAvg_1/2737238*
_output_shapes	
:А2 
b_norm_3/AssignMovingAvg_1/mul┼
.b_norm_3/AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOp"b_norm_3_assignmovingavg_1_2737238"b_norm_3/AssignMovingAvg_1/mul:z:0*^b_norm_3/AssignMovingAvg_1/ReadVariableOp*5
_class+
)'loc:@b_norm_3/AssignMovingAvg_1/2737238*
_output_shapes
 *
dtype020
.b_norm_3/AssignMovingAvg_1/AssignSubVariableOp|
RELU_3/ReluRelub_norm_3/FusedBatchNormV3:y:0*
T0*0
_output_shapes
:         А2
RELU_3/Relu╚
max_pooling2d_58/MaxPoolMaxPoolRELU_3/Relu:activations:0*0
_output_shapes
:         А*
ksize
*
paddingVALID*
strides
2
max_pooling2d_58/MaxPoolм
conv_4/Conv2D/ReadVariableOpReadVariableOp%conv_4_conv2d_readvariableop_resource*(
_output_shapes
:АА*
dtype02
conv_4/Conv2D/ReadVariableOp╘
conv_4/Conv2DConv2D!max_pooling2d_58/MaxPool:output:0$conv_4/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:         А*
paddingSAME*
strides
2
conv_4/Conv2Dв
conv_4/BiasAdd/ReadVariableOpReadVariableOp&conv_4_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02
conv_4/BiasAdd/ReadVariableOpе
conv_4/BiasAddBiasAddconv_4/Conv2D:output:0%conv_4/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:         А2
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
b_norm_4/LogicalAnd/yР
b_norm_4/LogicalAnd
LogicalAndb_norm_4/LogicalAnd/x:output:0b_norm_4/LogicalAnd/y:output:0*
_output_shapes
: 2
b_norm_4/LogicalAndР
b_norm_4/ReadVariableOpReadVariableOp b_norm_4_readvariableop_resource*
_output_shapes	
:А*
dtype02
b_norm_4/ReadVariableOpЦ
b_norm_4/ReadVariableOp_1ReadVariableOp"b_norm_4_readvariableop_1_resource*
_output_shapes	
:А*
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
b_norm_4/Const_1╤
b_norm_4/FusedBatchNormV3FusedBatchNormV3conv_4/BiasAdd:output:0b_norm_4/ReadVariableOp:value:0!b_norm_4/ReadVariableOp_1:value:0b_norm_4/Const:output:0b_norm_4/Const_1:output:0*
T0*
U0*P
_output_shapes>
<:         А:А:А:А:А:*
epsilon%oГ:2
b_norm_4/FusedBatchNormV3i
b_norm_4/Const_2Const*
_output_shapes
: *
dtype0*
valueB
 *дp}?2
b_norm_4/Const_2║
b_norm_4/AssignMovingAvg/sub/xConst*3
_class)
'%loc:@b_norm_4/AssignMovingAvg/2737269*
_output_shapes
: *
dtype0*
valueB
 *  А?2 
b_norm_4/AssignMovingAvg/sub/x▌
b_norm_4/AssignMovingAvg/subSub'b_norm_4/AssignMovingAvg/sub/x:output:0b_norm_4/Const_2:output:0*
T0*3
_class)
'%loc:@b_norm_4/AssignMovingAvg/2737269*
_output_shapes
: 2
b_norm_4/AssignMovingAvg/sub░
'b_norm_4/AssignMovingAvg/ReadVariableOpReadVariableOp b_norm_4_assignmovingavg_2737269*
_output_shapes	
:А*
dtype02)
'b_norm_4/AssignMovingAvg/ReadVariableOp√
b_norm_4/AssignMovingAvg/sub_1Sub/b_norm_4/AssignMovingAvg/ReadVariableOp:value:0&b_norm_4/FusedBatchNormV3:batch_mean:0*
T0*3
_class)
'%loc:@b_norm_4/AssignMovingAvg/2737269*
_output_shapes	
:А2 
b_norm_4/AssignMovingAvg/sub_1ф
b_norm_4/AssignMovingAvg/mulMul"b_norm_4/AssignMovingAvg/sub_1:z:0 b_norm_4/AssignMovingAvg/sub:z:0*
T0*3
_class)
'%loc:@b_norm_4/AssignMovingAvg/2737269*
_output_shapes	
:А2
b_norm_4/AssignMovingAvg/mul╣
,b_norm_4/AssignMovingAvg/AssignSubVariableOpAssignSubVariableOp b_norm_4_assignmovingavg_2737269 b_norm_4/AssignMovingAvg/mul:z:0(^b_norm_4/AssignMovingAvg/ReadVariableOp*3
_class)
'%loc:@b_norm_4/AssignMovingAvg/2737269*
_output_shapes
 *
dtype02.
,b_norm_4/AssignMovingAvg/AssignSubVariableOp└
 b_norm_4/AssignMovingAvg_1/sub/xConst*5
_class+
)'loc:@b_norm_4/AssignMovingAvg_1/2737276*
_output_shapes
: *
dtype0*
valueB
 *  А?2"
 b_norm_4/AssignMovingAvg_1/sub/xх
b_norm_4/AssignMovingAvg_1/subSub)b_norm_4/AssignMovingAvg_1/sub/x:output:0b_norm_4/Const_2:output:0*
T0*5
_class+
)'loc:@b_norm_4/AssignMovingAvg_1/2737276*
_output_shapes
: 2 
b_norm_4/AssignMovingAvg_1/sub╢
)b_norm_4/AssignMovingAvg_1/ReadVariableOpReadVariableOp"b_norm_4_assignmovingavg_1_2737276*
_output_shapes	
:А*
dtype02+
)b_norm_4/AssignMovingAvg_1/ReadVariableOpЗ
 b_norm_4/AssignMovingAvg_1/sub_1Sub1b_norm_4/AssignMovingAvg_1/ReadVariableOp:value:0*b_norm_4/FusedBatchNormV3:batch_variance:0*
T0*5
_class+
)'loc:@b_norm_4/AssignMovingAvg_1/2737276*
_output_shapes	
:А2"
 b_norm_4/AssignMovingAvg_1/sub_1ю
b_norm_4/AssignMovingAvg_1/mulMul$b_norm_4/AssignMovingAvg_1/sub_1:z:0"b_norm_4/AssignMovingAvg_1/sub:z:0*
T0*5
_class+
)'loc:@b_norm_4/AssignMovingAvg_1/2737276*
_output_shapes	
:А2 
b_norm_4/AssignMovingAvg_1/mul┼
.b_norm_4/AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOp"b_norm_4_assignmovingavg_1_2737276"b_norm_4/AssignMovingAvg_1/mul:z:0*^b_norm_4/AssignMovingAvg_1/ReadVariableOp*5
_class+
)'loc:@b_norm_4/AssignMovingAvg_1/2737276*
_output_shapes
 *
dtype020
.b_norm_4/AssignMovingAvg_1/AssignSubVariableOp|
RELU_4/ReluRelub_norm_4/FusedBatchNormV3:y:0*
T0*0
_output_shapes
:         А2
RELU_4/Relu╚
max_pooling2d_59/MaxPoolMaxPoolRELU_4/Relu:activations:0*0
_output_shapes
:         А*
ksize
*
paddingVALID*
strides
2
max_pooling2d_59/MaxPoolu
flatten_14/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2
flatten_14/Constд
flatten_14/ReshapeReshape!max_pooling2d_59/MaxPool:output:0flatten_14/Const:output:0*
T0*(
_output_shapes
:         А2
flatten_14/Reshapew
dropout_29/dropout/rateConst*
_output_shapes
: *
dtype0*
valueB
 *  А>2
dropout_29/dropout/rate
dropout_29/dropout/ShapeShapeflatten_14/Reshape:output:0*
T0*
_output_shapes
:2
dropout_29/dropout/ShapeУ
%dropout_29/dropout/random_uniform/minConst*
_output_shapes
: *
dtype0*
valueB
 *    2'
%dropout_29/dropout/random_uniform/minУ
%dropout_29/dropout/random_uniform/maxConst*
_output_shapes
: *
dtype0*
valueB
 *  А?2'
%dropout_29/dropout/random_uniform/max╓
/dropout_29/dropout/random_uniform/RandomUniformRandomUniform!dropout_29/dropout/Shape:output:0*
T0*(
_output_shapes
:         А*
dtype021
/dropout_29/dropout/random_uniform/RandomUniform╓
%dropout_29/dropout/random_uniform/subSub.dropout_29/dropout/random_uniform/max:output:0.dropout_29/dropout/random_uniform/min:output:0*
T0*
_output_shapes
: 2'
%dropout_29/dropout/random_uniform/subэ
%dropout_29/dropout/random_uniform/mulMul8dropout_29/dropout/random_uniform/RandomUniform:output:0)dropout_29/dropout/random_uniform/sub:z:0*
T0*(
_output_shapes
:         А2'
%dropout_29/dropout/random_uniform/mul█
!dropout_29/dropout/random_uniformAdd)dropout_29/dropout/random_uniform/mul:z:0.dropout_29/dropout/random_uniform/min:output:0*
T0*(
_output_shapes
:         А2#
!dropout_29/dropout/random_uniformy
dropout_29/dropout/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  А?2
dropout_29/dropout/sub/xЭ
dropout_29/dropout/subSub!dropout_29/dropout/sub/x:output:0 dropout_29/dropout/rate:output:0*
T0*
_output_shapes
: 2
dropout_29/dropout/subБ
dropout_29/dropout/truediv/xConst*
_output_shapes
: *
dtype0*
valueB
 *  А?2
dropout_29/dropout/truediv/xз
dropout_29/dropout/truedivRealDiv%dropout_29/dropout/truediv/x:output:0dropout_29/dropout/sub:z:0*
T0*
_output_shapes
: 2
dropout_29/dropout/truediv╬
dropout_29/dropout/GreaterEqualGreaterEqual%dropout_29/dropout/random_uniform:z:0 dropout_29/dropout/rate:output:0*
T0*(
_output_shapes
:         А2!
dropout_29/dropout/GreaterEqualз
dropout_29/dropout/mulMulflatten_14/Reshape:output:0dropout_29/dropout/truediv:z:0*
T0*(
_output_shapes
:         А2
dropout_29/dropout/mulб
dropout_29/dropout/CastCast#dropout_29/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:         А2
dropout_29/dropout/Castз
dropout_29/dropout/mul_1Muldropout_29/dropout/mul:z:0dropout_29/dropout/Cast:y:0*
T0*(
_output_shapes
:         А2
dropout_29/dropout/mul_1▌
/output_class_distribution/MatMul/ReadVariableOpReadVariableOp8output_class_distribution_matmul_readvariableop_resource* 
_output_shapes
:
А╚*
dtype021
/output_class_distribution/MatMul/ReadVariableOp╪
 output_class_distribution/MatMulMatMuldropout_29/dropout/mul_1:z:07output_class_distribution/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ╚2"
 output_class_distribution/MatMul█
0output_class_distribution/BiasAdd/ReadVariableOpReadVariableOp9output_class_distribution_biasadd_readvariableop_resource*
_output_shapes	
:╚*
dtype022
0output_class_distribution/BiasAdd/ReadVariableOpъ
!output_class_distribution/BiasAddBiasAdd*output_class_distribution/MatMul:product:08output_class_distribution/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ╚2#
!output_class_distribution/BiasAddй
IdentityIdentity*output_class_distribution/BiasAdd:output:0-^b_norm_1/AssignMovingAvg/AssignSubVariableOp(^b_norm_1/AssignMovingAvg/ReadVariableOp/^b_norm_1/AssignMovingAvg_1/AssignSubVariableOp*^b_norm_1/AssignMovingAvg_1/ReadVariableOp^b_norm_1/ReadVariableOp^b_norm_1/ReadVariableOp_1-^b_norm_2/AssignMovingAvg/AssignSubVariableOp(^b_norm_2/AssignMovingAvg/ReadVariableOp/^b_norm_2/AssignMovingAvg_1/AssignSubVariableOp*^b_norm_2/AssignMovingAvg_1/ReadVariableOp^b_norm_2/ReadVariableOp^b_norm_2/ReadVariableOp_1-^b_norm_3/AssignMovingAvg/AssignSubVariableOp(^b_norm_3/AssignMovingAvg/ReadVariableOp/^b_norm_3/AssignMovingAvg_1/AssignSubVariableOp*^b_norm_3/AssignMovingAvg_1/ReadVariableOp^b_norm_3/ReadVariableOp^b_norm_3/ReadVariableOp_1-^b_norm_4/AssignMovingAvg/AssignSubVariableOp(^b_norm_4/AssignMovingAvg/ReadVariableOp/^b_norm_4/AssignMovingAvg_1/AssignSubVariableOp*^b_norm_4/AssignMovingAvg_1/ReadVariableOp^b_norm_4/ReadVariableOp^b_norm_4/ReadVariableOp_1^conv_1/BiasAdd/ReadVariableOp^conv_1/Conv2D/ReadVariableOp^conv_2/BiasAdd/ReadVariableOp^conv_2/Conv2D/ReadVariableOp^conv_3/BiasAdd/ReadVariableOp^conv_3/Conv2D/ReadVariableOp^conv_4/BiasAdd/ReadVariableOp^conv_4/Conv2D/ReadVariableOp>^input_batch_normalization/AssignMovingAvg/AssignSubVariableOp9^input_batch_normalization/AssignMovingAvg/ReadVariableOp@^input_batch_normalization/AssignMovingAvg_1/AssignSubVariableOp;^input_batch_normalization/AssignMovingAvg_1/ReadVariableOp)^input_batch_normalization/ReadVariableOp+^input_batch_normalization/ReadVariableOp_1"^input_conv/BiasAdd/ReadVariableOp!^input_conv/Conv2D/ReadVariableOp1^output_class_distribution/BiasAdd/ReadVariableOp0^output_class_distribution/MatMul/ReadVariableOp*
T0*(
_output_shapes
:         ╚2

Identity"
identityIdentity:output:0*▓
_input_shapesа
Э:         АА::::::::::::::::::::::::::::::::2\
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
8input_batch_normalization/AssignMovingAvg/ReadVariableOp8input_batch_normalization/AssignMovingAvg/ReadVariableOp2В
?input_batch_normalization/AssignMovingAvg_1/AssignSubVariableOp?input_batch_normalization/AssignMovingAvg_1/AssignSubVariableOp2x
:input_batch_normalization/AssignMovingAvg_1/ReadVariableOp:input_batch_normalization/AssignMovingAvg_1/ReadVariableOp2T
(input_batch_normalization/ReadVariableOp(input_batch_normalization/ReadVariableOp2X
*input_batch_normalization/ReadVariableOp_1*input_batch_normalization/ReadVariableOp_12F
!input_conv/BiasAdd/ReadVariableOp!input_conv/BiasAdd/ReadVariableOp2D
 input_conv/Conv2D/ReadVariableOp input_conv/Conv2D/ReadVariableOp2d
0output_class_distribution/BiasAdd/ReadVariableOp0output_class_distribution/BiasAdd/ReadVariableOp2b
/output_class_distribution/MatMul/ReadVariableOp/output_class_distribution/MatMul/ReadVariableOp:& "
 
_user_specified_nameinputs
╫
є
*__inference_b_norm_1_layer_call_fn_2737884

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4
identityИвStatefulPartitionedCallц
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4*
Tin	
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*A
_output_shapes/
-:+                            *-
config_proto

GPU

CPU2*0J 8*N
fIRG
E__inference_b_norm_1_layer_call_and_return_conditional_losses_27355762
StatefulPartitionedCallи
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+                            2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+                            ::::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs
╫
є
*__inference_b_norm_2_layer_call_fn_2737980

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4
identityИвStatefulPartitionedCallц
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4*
Tin	
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*A
_output_shapes/
-:+                           @*-
config_proto

GPU

CPU2*0J 8*N
fIRG
E__inference_b_norm_2_layer_call_and_return_conditional_losses_27357402
StatefulPartitionedCallи
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+                           @2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+                           @::::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs
д
e
G__inference_dropout_29_layer_call_and_return_conditional_losses_2738449

inputs

identity_1[
IdentityIdentityinputs*
T0*(
_output_shapes
:         А2

Identityj

Identity_1IdentityIdentity:output:0*
T0*(
_output_shapes
:         А2

Identity_1"!

identity_1Identity_1:output:0*'
_input_shapes
:         А:& "
 
_user_specified_nameinputs
д
є
*__inference_b_norm_3_layer_call_fn_2738224

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4
identityИвStatefulPartitionedCall╒
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4*
Tin	
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*0
_output_shapes
:         А*-
config_proto

GPU

CPU2*0J 8*N
fIRG
E__inference_b_norm_3_layer_call_and_return_conditional_losses_27364842
StatefulPartitionedCallЧ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*0
_output_shapes
:         А2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:         А::::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs
┌
є
*__inference_b_norm_3_layer_call_fn_2738150

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4
identityИвStatefulPartitionedCallч
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4*
Tin	
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*B
_output_shapes0
.:,                           А*-
config_proto

GPU

CPU2*0J 8*N
fIRG
E__inference_b_norm_3_layer_call_and_return_conditional_losses_27359042
StatefulPartitionedCallй
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*B
_output_shapes0
.:,                           А2

Identity"
identityIdentity:output:0*Q
_input_shapes@
>:,                           А::::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs
╖
i
M__inference_max_pooling2d_60_layer_call_and_return_conditional_losses_2735456

inputs
identityн
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4                                    *
ksize
*
paddingVALID*
strides
2	
MaxPoolЗ
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4                                    2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4                                    :& "
 
_user_specified_nameinputs
а
_
C__inference_RELU_2_layer_call_and_return_conditional_losses_2738068

inputs
identityV
ReluReluinputs*
T0*/
_output_shapes
:           @2
Relun
IdentityIdentityRelu:activations:0*
T0*/
_output_shapes
:           @2

Identity"
identityIdentity:output:0*.
_input_shapes
:           @:& "
 
_user_specified_nameinputs
╩
ш
E__inference_b_norm_4_layer_call_and_return_conditional_losses_2738311

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityИвFusedBatchNormV3/ReadVariableOpв!FusedBatchNormV3/ReadVariableOp_1вReadVariableOpвReadVariableOp_1^
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

LogicalAndu
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:А*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:А*
dtype02
ReadVariableOp_1и
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:А*
dtype02!
FusedBatchNormV3/ReadVariableOpо
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:А*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1╧
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:         А:А:А:А:А:*
epsilon%oГ:*
is_training( 2
FusedBatchNormV3S
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *дp}?2
Const█
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*0
_output_shapes
:         А2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:         А::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:& "
 
_user_specified_nameinputs
Ч$
Т
E__inference_b_norm_3_layer_call_and_return_conditional_losses_2738193

inputs
readvariableop_resource
readvariableop_1_resource
assignmovingavg_2738178
assignmovingavg_1_2738185
identityИв#AssignMovingAvg/AssignSubVariableOpвAssignMovingAvg/ReadVariableOpв%AssignMovingAvg_1/AssignSubVariableOpв AssignMovingAvg_1/ReadVariableOpвReadVariableOpвReadVariableOp_1^
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

LogicalAndu
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:А*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:А*
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
Const_1К
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0Const:output:0Const_1:output:0*
T0*
U0*P
_output_shapes>
<:         А:А:А:А:А:*
epsilon%oГ:2
FusedBatchNormV3W
Const_2Const*
_output_shapes
: *
dtype0*
valueB
 *дp}?2	
Const_2Я
AssignMovingAvg/sub/xConst**
_class 
loc:@AssignMovingAvg/2738178*
_output_shapes
: *
dtype0*
valueB
 *  А?2
AssignMovingAvg/sub/x░
AssignMovingAvg/subSubAssignMovingAvg/sub/x:output:0Const_2:output:0*
T0**
_class 
loc:@AssignMovingAvg/2738178*
_output_shapes
: 2
AssignMovingAvg/subХ
AssignMovingAvg/ReadVariableOpReadVariableOpassignmovingavg_2738178*
_output_shapes	
:А*
dtype02 
AssignMovingAvg/ReadVariableOp╬
AssignMovingAvg/sub_1Sub&AssignMovingAvg/ReadVariableOp:value:0FusedBatchNormV3:batch_mean:0*
T0**
_class 
loc:@AssignMovingAvg/2738178*
_output_shapes	
:А2
AssignMovingAvg/sub_1╖
AssignMovingAvg/mulMulAssignMovingAvg/sub_1:z:0AssignMovingAvg/sub:z:0*
T0**
_class 
loc:@AssignMovingAvg/2738178*
_output_shapes	
:А2
AssignMovingAvg/mulГ
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOpassignmovingavg_2738178AssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp**
_class 
loc:@AssignMovingAvg/2738178*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOpе
AssignMovingAvg_1/sub/xConst*,
_class"
 loc:@AssignMovingAvg_1/2738185*
_output_shapes
: *
dtype0*
valueB
 *  А?2
AssignMovingAvg_1/sub/x╕
AssignMovingAvg_1/subSub AssignMovingAvg_1/sub/x:output:0Const_2:output:0*
T0*,
_class"
 loc:@AssignMovingAvg_1/2738185*
_output_shapes
: 2
AssignMovingAvg_1/subЫ
 AssignMovingAvg_1/ReadVariableOpReadVariableOpassignmovingavg_1_2738185*
_output_shapes	
:А*
dtype02"
 AssignMovingAvg_1/ReadVariableOp┌
AssignMovingAvg_1/sub_1Sub(AssignMovingAvg_1/ReadVariableOp:value:0!FusedBatchNormV3:batch_variance:0*
T0*,
_class"
 loc:@AssignMovingAvg_1/2738185*
_output_shapes	
:А2
AssignMovingAvg_1/sub_1┴
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub_1:z:0AssignMovingAvg_1/sub:z:0*
T0*,
_class"
 loc:@AssignMovingAvg_1/2738185*
_output_shapes	
:А2
AssignMovingAvg_1/mulП
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOpassignmovingavg_1_2738185AssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*,
_class"
 loc:@AssignMovingAvg_1/2738185*
_output_shapes
 *
dtype02'
%AssignMovingAvg_1/AssignSubVariableOpз
IdentityIdentityFusedBatchNormV3:y:0$^AssignMovingAvg/AssignSubVariableOp^AssignMovingAvg/ReadVariableOp&^AssignMovingAvg_1/AssignSubVariableOp!^AssignMovingAvg_1/ReadVariableOp^ReadVariableOp^ReadVariableOp_1*
T0*0
_output_shapes
:         А2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:         А::::2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp2D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp2 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:& "
 
_user_specified_nameinputs
Ч$
Т
E__inference_b_norm_4_layer_call_and_return_conditional_losses_2736580

inputs
readvariableop_resource
readvariableop_1_resource
assignmovingavg_2736565
assignmovingavg_1_2736572
identityИв#AssignMovingAvg/AssignSubVariableOpвAssignMovingAvg/ReadVariableOpв%AssignMovingAvg_1/AssignSubVariableOpв AssignMovingAvg_1/ReadVariableOpвReadVariableOpвReadVariableOp_1^
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

LogicalAndu
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:А*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:А*
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
Const_1К
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0Const:output:0Const_1:output:0*
T0*
U0*P
_output_shapes>
<:         А:А:А:А:А:*
epsilon%oГ:2
FusedBatchNormV3W
Const_2Const*
_output_shapes
: *
dtype0*
valueB
 *дp}?2	
Const_2Я
AssignMovingAvg/sub/xConst**
_class 
loc:@AssignMovingAvg/2736565*
_output_shapes
: *
dtype0*
valueB
 *  А?2
AssignMovingAvg/sub/x░
AssignMovingAvg/subSubAssignMovingAvg/sub/x:output:0Const_2:output:0*
T0**
_class 
loc:@AssignMovingAvg/2736565*
_output_shapes
: 2
AssignMovingAvg/subХ
AssignMovingAvg/ReadVariableOpReadVariableOpassignmovingavg_2736565*
_output_shapes	
:А*
dtype02 
AssignMovingAvg/ReadVariableOp╬
AssignMovingAvg/sub_1Sub&AssignMovingAvg/ReadVariableOp:value:0FusedBatchNormV3:batch_mean:0*
T0**
_class 
loc:@AssignMovingAvg/2736565*
_output_shapes	
:А2
AssignMovingAvg/sub_1╖
AssignMovingAvg/mulMulAssignMovingAvg/sub_1:z:0AssignMovingAvg/sub:z:0*
T0**
_class 
loc:@AssignMovingAvg/2736565*
_output_shapes	
:А2
AssignMovingAvg/mulГ
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOpassignmovingavg_2736565AssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp**
_class 
loc:@AssignMovingAvg/2736565*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOpе
AssignMovingAvg_1/sub/xConst*,
_class"
 loc:@AssignMovingAvg_1/2736572*
_output_shapes
: *
dtype0*
valueB
 *  А?2
AssignMovingAvg_1/sub/x╕
AssignMovingAvg_1/subSub AssignMovingAvg_1/sub/x:output:0Const_2:output:0*
T0*,
_class"
 loc:@AssignMovingAvg_1/2736572*
_output_shapes
: 2
AssignMovingAvg_1/subЫ
 AssignMovingAvg_1/ReadVariableOpReadVariableOpassignmovingavg_1_2736572*
_output_shapes	
:А*
dtype02"
 AssignMovingAvg_1/ReadVariableOp┌
AssignMovingAvg_1/sub_1Sub(AssignMovingAvg_1/ReadVariableOp:value:0!FusedBatchNormV3:batch_variance:0*
T0*,
_class"
 loc:@AssignMovingAvg_1/2736572*
_output_shapes	
:А2
AssignMovingAvg_1/sub_1┴
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub_1:z:0AssignMovingAvg_1/sub:z:0*
T0*,
_class"
 loc:@AssignMovingAvg_1/2736572*
_output_shapes	
:А2
AssignMovingAvg_1/mulП
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOpassignmovingavg_1_2736572AssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*,
_class"
 loc:@AssignMovingAvg_1/2736572*
_output_shapes
 *
dtype02'
%AssignMovingAvg_1/AssignSubVariableOpз
IdentityIdentityFusedBatchNormV3:y:0$^AssignMovingAvg/AssignSubVariableOp^AssignMovingAvg/ReadVariableOp&^AssignMovingAvg_1/AssignSubVariableOp!^AssignMovingAvg_1/ReadVariableOp^ReadVariableOp^ReadVariableOp_1*
T0*0
_output_shapes
:         А2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:         А::::2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp2D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp2 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:& "
 
_user_specified_nameinputs
╛$
Т
E__inference_b_norm_2_layer_call_and_return_conditional_losses_2737949

inputs
readvariableop_resource
readvariableop_1_resource
assignmovingavg_2737934
assignmovingavg_1_2737941
identityИв#AssignMovingAvg/AssignSubVariableOpвAssignMovingAvg/ReadVariableOpв%AssignMovingAvg_1/AssignSubVariableOpв AssignMovingAvg_1/ReadVariableOpвReadVariableOpвReadVariableOp_1^
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
Const_1Ч
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0Const:output:0Const_1:output:0*
T0*
U0*]
_output_shapesK
I:+                           @:@:@:@:@:*
epsilon%oГ:2
FusedBatchNormV3W
Const_2Const*
_output_shapes
: *
dtype0*
valueB
 *дp}?2	
Const_2Я
AssignMovingAvg/sub/xConst**
_class 
loc:@AssignMovingAvg/2737934*
_output_shapes
: *
dtype0*
valueB
 *  А?2
AssignMovingAvg/sub/x░
AssignMovingAvg/subSubAssignMovingAvg/sub/x:output:0Const_2:output:0*
T0**
_class 
loc:@AssignMovingAvg/2737934*
_output_shapes
: 2
AssignMovingAvg/subФ
AssignMovingAvg/ReadVariableOpReadVariableOpassignmovingavg_2737934*
_output_shapes
:@*
dtype02 
AssignMovingAvg/ReadVariableOp═
AssignMovingAvg/sub_1Sub&AssignMovingAvg/ReadVariableOp:value:0FusedBatchNormV3:batch_mean:0*
T0**
_class 
loc:@AssignMovingAvg/2737934*
_output_shapes
:@2
AssignMovingAvg/sub_1╢
AssignMovingAvg/mulMulAssignMovingAvg/sub_1:z:0AssignMovingAvg/sub:z:0*
T0**
_class 
loc:@AssignMovingAvg/2737934*
_output_shapes
:@2
AssignMovingAvg/mulГ
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOpassignmovingavg_2737934AssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp**
_class 
loc:@AssignMovingAvg/2737934*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOpе
AssignMovingAvg_1/sub/xConst*,
_class"
 loc:@AssignMovingAvg_1/2737941*
_output_shapes
: *
dtype0*
valueB
 *  А?2
AssignMovingAvg_1/sub/x╕
AssignMovingAvg_1/subSub AssignMovingAvg_1/sub/x:output:0Const_2:output:0*
T0*,
_class"
 loc:@AssignMovingAvg_1/2737941*
_output_shapes
: 2
AssignMovingAvg_1/subЪ
 AssignMovingAvg_1/ReadVariableOpReadVariableOpassignmovingavg_1_2737941*
_output_shapes
:@*
dtype02"
 AssignMovingAvg_1/ReadVariableOp┘
AssignMovingAvg_1/sub_1Sub(AssignMovingAvg_1/ReadVariableOp:value:0!FusedBatchNormV3:batch_variance:0*
T0*,
_class"
 loc:@AssignMovingAvg_1/2737941*
_output_shapes
:@2
AssignMovingAvg_1/sub_1└
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub_1:z:0AssignMovingAvg_1/sub:z:0*
T0*,
_class"
 loc:@AssignMovingAvg_1/2737941*
_output_shapes
:@2
AssignMovingAvg_1/mulП
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOpassignmovingavg_1_2737941AssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*,
_class"
 loc:@AssignMovingAvg_1/2737941*
_output_shapes
 *
dtype02'
%AssignMovingAvg_1/AssignSubVariableOp╕
IdentityIdentityFusedBatchNormV3:y:0$^AssignMovingAvg/AssignSubVariableOp^AssignMovingAvg/ReadVariableOp&^AssignMovingAvg_1/AssignSubVariableOp!^AssignMovingAvg_1/ReadVariableOp^ReadVariableOp^ReadVariableOp_1*
T0*A
_output_shapes/
-:+                           @2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+                           @::::2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp2D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp2 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:& "
 
_user_specified_nameinputs
╧$
г
V__inference_input_batch_normalization_layer_call_and_return_conditional_losses_2735412

inputs
readvariableop_resource
readvariableop_1_resource
assignmovingavg_2735397
assignmovingavg_1_2735404
identityИв#AssignMovingAvg/AssignSubVariableOpвAssignMovingAvg/ReadVariableOpв%AssignMovingAvg_1/AssignSubVariableOpв AssignMovingAvg_1/ReadVariableOpвReadVariableOpвReadVariableOp_1^
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
Const_1Ч
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0Const:output:0Const_1:output:0*
T0*
U0*]
_output_shapesK
I:+                            : : : : :*
epsilon%oГ:2
FusedBatchNormV3W
Const_2Const*
_output_shapes
: *
dtype0*
valueB
 *дp}?2	
Const_2Я
AssignMovingAvg/sub/xConst**
_class 
loc:@AssignMovingAvg/2735397*
_output_shapes
: *
dtype0*
valueB
 *  А?2
AssignMovingAvg/sub/x░
AssignMovingAvg/subSubAssignMovingAvg/sub/x:output:0Const_2:output:0*
T0**
_class 
loc:@AssignMovingAvg/2735397*
_output_shapes
: 2
AssignMovingAvg/subФ
AssignMovingAvg/ReadVariableOpReadVariableOpassignmovingavg_2735397*
_output_shapes
: *
dtype02 
AssignMovingAvg/ReadVariableOp═
AssignMovingAvg/sub_1Sub&AssignMovingAvg/ReadVariableOp:value:0FusedBatchNormV3:batch_mean:0*
T0**
_class 
loc:@AssignMovingAvg/2735397*
_output_shapes
: 2
AssignMovingAvg/sub_1╢
AssignMovingAvg/mulMulAssignMovingAvg/sub_1:z:0AssignMovingAvg/sub:z:0*
T0**
_class 
loc:@AssignMovingAvg/2735397*
_output_shapes
: 2
AssignMovingAvg/mulГ
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOpassignmovingavg_2735397AssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp**
_class 
loc:@AssignMovingAvg/2735397*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOpе
AssignMovingAvg_1/sub/xConst*,
_class"
 loc:@AssignMovingAvg_1/2735404*
_output_shapes
: *
dtype0*
valueB
 *  А?2
AssignMovingAvg_1/sub/x╕
AssignMovingAvg_1/subSub AssignMovingAvg_1/sub/x:output:0Const_2:output:0*
T0*,
_class"
 loc:@AssignMovingAvg_1/2735404*
_output_shapes
: 2
AssignMovingAvg_1/subЪ
 AssignMovingAvg_1/ReadVariableOpReadVariableOpassignmovingavg_1_2735404*
_output_shapes
: *
dtype02"
 AssignMovingAvg_1/ReadVariableOp┘
AssignMovingAvg_1/sub_1Sub(AssignMovingAvg_1/ReadVariableOp:value:0!FusedBatchNormV3:batch_variance:0*
T0*,
_class"
 loc:@AssignMovingAvg_1/2735404*
_output_shapes
: 2
AssignMovingAvg_1/sub_1└
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub_1:z:0AssignMovingAvg_1/sub:z:0*
T0*,
_class"
 loc:@AssignMovingAvg_1/2735404*
_output_shapes
: 2
AssignMovingAvg_1/mulП
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOpassignmovingavg_1_2735404AssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*,
_class"
 loc:@AssignMovingAvg_1/2735404*
_output_shapes
 *
dtype02'
%AssignMovingAvg_1/AssignSubVariableOp╕
IdentityIdentityFusedBatchNormV3:y:0$^AssignMovingAvg/AssignSubVariableOp^AssignMovingAvg/ReadVariableOp&^AssignMovingAvg_1/AssignSubVariableOp!^AssignMovingAvg_1/ReadVariableOp^ReadVariableOp^ReadVariableOp_1*
T0*A
_output_shapes/
-:+                            2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+                            ::::2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp2D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp2 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:& "
 
_user_specified_nameinputs
╥
N
2__inference_max_pooling2d_58_layer_call_fn_2735954

inputs
identity█
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*J
_output_shapes8
6:4                                    *-
config_proto

GPU

CPU2*0J 8*V
fQRO
M__inference_max_pooling2d_58_layer_call_and_return_conditional_losses_27359482
PartitionedCallП
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4                                    2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4                                    :& "
 
_user_specified_nameinputs
═$
Т
E__inference_b_norm_4_layer_call_and_return_conditional_losses_2738363

inputs
readvariableop_resource
readvariableop_1_resource
assignmovingavg_2738348
assignmovingavg_1_2738355
identityИв#AssignMovingAvg/AssignSubVariableOpвAssignMovingAvg/ReadVariableOpв%AssignMovingAvg_1/AssignSubVariableOpв AssignMovingAvg_1/ReadVariableOpвReadVariableOpвReadVariableOp_1^
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

LogicalAndu
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:А*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:А*
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
Const_1Ь
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0Const:output:0Const_1:output:0*
T0*
U0*b
_output_shapesP
N:,                           А:А:А:А:А:*
epsilon%oГ:2
FusedBatchNormV3W
Const_2Const*
_output_shapes
: *
dtype0*
valueB
 *дp}?2	
Const_2Я
AssignMovingAvg/sub/xConst**
_class 
loc:@AssignMovingAvg/2738348*
_output_shapes
: *
dtype0*
valueB
 *  А?2
AssignMovingAvg/sub/x░
AssignMovingAvg/subSubAssignMovingAvg/sub/x:output:0Const_2:output:0*
T0**
_class 
loc:@AssignMovingAvg/2738348*
_output_shapes
: 2
AssignMovingAvg/subХ
AssignMovingAvg/ReadVariableOpReadVariableOpassignmovingavg_2738348*
_output_shapes	
:А*
dtype02 
AssignMovingAvg/ReadVariableOp╬
AssignMovingAvg/sub_1Sub&AssignMovingAvg/ReadVariableOp:value:0FusedBatchNormV3:batch_mean:0*
T0**
_class 
loc:@AssignMovingAvg/2738348*
_output_shapes	
:А2
AssignMovingAvg/sub_1╖
AssignMovingAvg/mulMulAssignMovingAvg/sub_1:z:0AssignMovingAvg/sub:z:0*
T0**
_class 
loc:@AssignMovingAvg/2738348*
_output_shapes	
:А2
AssignMovingAvg/mulГ
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOpassignmovingavg_2738348AssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp**
_class 
loc:@AssignMovingAvg/2738348*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOpе
AssignMovingAvg_1/sub/xConst*,
_class"
 loc:@AssignMovingAvg_1/2738355*
_output_shapes
: *
dtype0*
valueB
 *  А?2
AssignMovingAvg_1/sub/x╕
AssignMovingAvg_1/subSub AssignMovingAvg_1/sub/x:output:0Const_2:output:0*
T0*,
_class"
 loc:@AssignMovingAvg_1/2738355*
_output_shapes
: 2
AssignMovingAvg_1/subЫ
 AssignMovingAvg_1/ReadVariableOpReadVariableOpassignmovingavg_1_2738355*
_output_shapes	
:А*
dtype02"
 AssignMovingAvg_1/ReadVariableOp┌
AssignMovingAvg_1/sub_1Sub(AssignMovingAvg_1/ReadVariableOp:value:0!FusedBatchNormV3:batch_variance:0*
T0*,
_class"
 loc:@AssignMovingAvg_1/2738355*
_output_shapes	
:А2
AssignMovingAvg_1/sub_1┴
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub_1:z:0AssignMovingAvg_1/sub:z:0*
T0*,
_class"
 loc:@AssignMovingAvg_1/2738355*
_output_shapes	
:А2
AssignMovingAvg_1/mulП
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOpassignmovingavg_1_2738355AssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*,
_class"
 loc:@AssignMovingAvg_1/2738355*
_output_shapes
 *
dtype02'
%AssignMovingAvg_1/AssignSubVariableOp╣
IdentityIdentityFusedBatchNormV3:y:0$^AssignMovingAvg/AssignSubVariableOp^AssignMovingAvg/ReadVariableOp&^AssignMovingAvg_1/AssignSubVariableOp!^AssignMovingAvg_1/ReadVariableOp^ReadVariableOp^ReadVariableOp_1*
T0*B
_output_shapes0
.:,                           А2

Identity"
identityIdentity:output:0*Q
_input_shapes@
>:,                           А::::2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp2D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp2 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:& "
 
_user_specified_nameinputs
а
_
C__inference_RELU_1_layer_call_and_return_conditional_losses_2736343

inputs
identityV
ReluReluinputs*
T0*/
_output_shapes
:         @@ 2
Relun
IdentityIdentityRelu:activations:0*
T0*/
_output_shapes
:         @@ 2

Identity"
identityIdentity:output:0*.
_input_shapes
:         @@ :& "
 
_user_specified_nameinputs
ф
Ж

,__inference_awsome_net_layer_call_fn_2736873
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
identityИвStatefulPartitionedCallМ
StatefulPartitionedCallStatefulPartitionedCallinput_conv_inputstatefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4statefulpartitionedcall_args_5statefulpartitionedcall_args_6statefulpartitionedcall_args_7statefulpartitionedcall_args_8statefulpartitionedcall_args_9statefulpartitionedcall_args_10statefulpartitionedcall_args_11statefulpartitionedcall_args_12statefulpartitionedcall_args_13statefulpartitionedcall_args_14statefulpartitionedcall_args_15statefulpartitionedcall_args_16statefulpartitionedcall_args_17statefulpartitionedcall_args_18statefulpartitionedcall_args_19statefulpartitionedcall_args_20statefulpartitionedcall_args_21statefulpartitionedcall_args_22statefulpartitionedcall_args_23statefulpartitionedcall_args_24statefulpartitionedcall_args_25statefulpartitionedcall_args_26statefulpartitionedcall_args_27statefulpartitionedcall_args_28statefulpartitionedcall_args_29statefulpartitionedcall_args_30statefulpartitionedcall_args_31statefulpartitionedcall_args_32*,
Tin%
#2!*
Tout
2*,
_gradient_op_typePartitionedCallUnused*(
_output_shapes
:         ╚*-
config_proto

GPU

CPU2*0J 8*P
fKRI
G__inference_awsome_net_layer_call_and_return_conditional_losses_27368382
StatefulPartitionedCallП
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:         ╚2

Identity"
identityIdentity:output:0*▓
_input_shapesа
Э:         АА::::::::::::::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:0 ,
*
_user_specified_nameinput_conv_input
Їo
¤
G__inference_awsome_net_layer_call_and_return_conditional_losses_2736935

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
identityИв b_norm_1/StatefulPartitionedCallв b_norm_2/StatefulPartitionedCallв b_norm_3/StatefulPartitionedCallв b_norm_4/StatefulPartitionedCallвconv_1/StatefulPartitionedCallвconv_2/StatefulPartitionedCallвconv_3/StatefulPartitionedCallвconv_4/StatefulPartitionedCallв1input_batch_normalization/StatefulPartitionedCallв"input_conv/StatefulPartitionedCallв1output_class_distribution/StatefulPartitionedCall┬
"input_conv/StatefulPartitionedCallStatefulPartitionedCallinputs)input_conv_statefulpartitionedcall_args_1)input_conv_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*1
_output_shapes
:         АА *-
config_proto

GPU

CPU2*0J 8*P
fKRI
G__inference_input_conv_layer_call_and_return_conditional_losses_27353102$
"input_conv/StatefulPartitionedCallи
1input_batch_normalization/StatefulPartitionedCallStatefulPartitionedCall+input_conv/StatefulPartitionedCall:output:08input_batch_normalization_statefulpartitionedcall_args_18input_batch_normalization_statefulpartitionedcall_args_28input_batch_normalization_statefulpartitionedcall_args_38input_batch_normalization_statefulpartitionedcall_args_4*
Tin	
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*1
_output_shapes
:         АА *-
config_proto

GPU

CPU2*0J 8*_
fZRX
V__inference_input_batch_normalization_layer_call_and_return_conditional_losses_273618023
1input_batch_normalization/StatefulPartitionedCallЖ
input_RELU/PartitionedCallPartitionedCall:input_batch_normalization/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*1
_output_shapes
:         АА *-
config_proto

GPU

CPU2*0J 8*P
fKRI
G__inference_input_RELU_layer_call_and_return_conditional_losses_27362092
input_RELU/PartitionedCall 
 max_pooling2d_60/PartitionedCallPartitionedCall#input_RELU/PartitionedCall:output:0*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*/
_output_shapes
:         @@ *-
config_proto

GPU

CPU2*0J 8*V
fQRO
M__inference_max_pooling2d_60_layer_call_and_return_conditional_losses_27354562"
 max_pooling2d_60/PartitionedCallє
dropout_28/PartitionedCallPartitionedCall)max_pooling2d_60/PartitionedCall:output:0*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*/
_output_shapes
:         @@ *-
config_proto

GPU

CPU2*0J 8*P
fKRI
G__inference_dropout_28_layer_call_and_return_conditional_losses_27362432
dropout_28/PartitionedCall╔
conv_1/StatefulPartitionedCallStatefulPartitionedCall#dropout_28/PartitionedCall:output:0%conv_1_statefulpartitionedcall_args_1%conv_1_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*/
_output_shapes
:         @@ *-
config_proto

GPU

CPU2*0J 8*L
fGRE
C__inference_conv_1_layer_call_and_return_conditional_losses_27354742 
conv_1/StatefulPartitionedCallл
 b_norm_1/StatefulPartitionedCallStatefulPartitionedCall'conv_1/StatefulPartitionedCall:output:0'b_norm_1_statefulpartitionedcall_args_1'b_norm_1_statefulpartitionedcall_args_2'b_norm_1_statefulpartitionedcall_args_3'b_norm_1_statefulpartitionedcall_args_4*
Tin	
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*/
_output_shapes
:         @@ *-
config_proto

GPU

CPU2*0J 8*N
fIRG
E__inference_b_norm_1_layer_call_and_return_conditional_losses_27363142"
 b_norm_1/StatefulPartitionedCallч
RELU_1/PartitionedCallPartitionedCall)b_norm_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*/
_output_shapes
:         @@ *-
config_proto

GPU

CPU2*0J 8*L
fGRE
C__inference_RELU_1_layer_call_and_return_conditional_losses_27363432
RELU_1/PartitionedCall√
 max_pooling2d_56/PartitionedCallPartitionedCallRELU_1/PartitionedCall:output:0*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*/
_output_shapes
:            *-
config_proto

GPU

CPU2*0J 8*V
fQRO
M__inference_max_pooling2d_56_layer_call_and_return_conditional_losses_27356202"
 max_pooling2d_56/PartitionedCall╧
conv_2/StatefulPartitionedCallStatefulPartitionedCall)max_pooling2d_56/PartitionedCall:output:0%conv_2_statefulpartitionedcall_args_1%conv_2_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*/
_output_shapes
:           @*-
config_proto

GPU

CPU2*0J 8*L
fGRE
C__inference_conv_2_layer_call_and_return_conditional_losses_27356382 
conv_2/StatefulPartitionedCallл
 b_norm_2/StatefulPartitionedCallStatefulPartitionedCall'conv_2/StatefulPartitionedCall:output:0'b_norm_2_statefulpartitionedcall_args_1'b_norm_2_statefulpartitionedcall_args_2'b_norm_2_statefulpartitionedcall_args_3'b_norm_2_statefulpartitionedcall_args_4*
Tin	
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*/
_output_shapes
:           @*-
config_proto

GPU

CPU2*0J 8*N
fIRG
E__inference_b_norm_2_layer_call_and_return_conditional_losses_27364102"
 b_norm_2/StatefulPartitionedCallч
RELU_2/PartitionedCallPartitionedCall)b_norm_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*/
_output_shapes
:           @*-
config_proto

GPU

CPU2*0J 8*L
fGRE
C__inference_RELU_2_layer_call_and_return_conditional_losses_27364392
RELU_2/PartitionedCall√
 max_pooling2d_57/PartitionedCallPartitionedCallRELU_2/PartitionedCall:output:0*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*/
_output_shapes
:         @*-
config_proto

GPU

CPU2*0J 8*V
fQRO
M__inference_max_pooling2d_57_layer_call_and_return_conditional_losses_27357842"
 max_pooling2d_57/PartitionedCall╨
conv_3/StatefulPartitionedCallStatefulPartitionedCall)max_pooling2d_57/PartitionedCall:output:0%conv_3_statefulpartitionedcall_args_1%conv_3_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*0
_output_shapes
:         А*-
config_proto

GPU

CPU2*0J 8*L
fGRE
C__inference_conv_3_layer_call_and_return_conditional_losses_27358022 
conv_3/StatefulPartitionedCallм
 b_norm_3/StatefulPartitionedCallStatefulPartitionedCall'conv_3/StatefulPartitionedCall:output:0'b_norm_3_statefulpartitionedcall_args_1'b_norm_3_statefulpartitionedcall_args_2'b_norm_3_statefulpartitionedcall_args_3'b_norm_3_statefulpartitionedcall_args_4*
Tin	
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*0
_output_shapes
:         А*-
config_proto

GPU

CPU2*0J 8*N
fIRG
E__inference_b_norm_3_layer_call_and_return_conditional_losses_27365062"
 b_norm_3/StatefulPartitionedCallш
RELU_3/PartitionedCallPartitionedCall)b_norm_3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*0
_output_shapes
:         А*-
config_proto

GPU

CPU2*0J 8*L
fGRE
C__inference_RELU_3_layer_call_and_return_conditional_losses_27365352
RELU_3/PartitionedCall№
 max_pooling2d_58/PartitionedCallPartitionedCallRELU_3/PartitionedCall:output:0*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*0
_output_shapes
:         А*-
config_proto

GPU

CPU2*0J 8*V
fQRO
M__inference_max_pooling2d_58_layer_call_and_return_conditional_losses_27359482"
 max_pooling2d_58/PartitionedCall╨
conv_4/StatefulPartitionedCallStatefulPartitionedCall)max_pooling2d_58/PartitionedCall:output:0%conv_4_statefulpartitionedcall_args_1%conv_4_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*0
_output_shapes
:         А*-
config_proto

GPU

CPU2*0J 8*L
fGRE
C__inference_conv_4_layer_call_and_return_conditional_losses_27359662 
conv_4/StatefulPartitionedCallм
 b_norm_4/StatefulPartitionedCallStatefulPartitionedCall'conv_4/StatefulPartitionedCall:output:0'b_norm_4_statefulpartitionedcall_args_1'b_norm_4_statefulpartitionedcall_args_2'b_norm_4_statefulpartitionedcall_args_3'b_norm_4_statefulpartitionedcall_args_4*
Tin	
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*0
_output_shapes
:         А*-
config_proto

GPU

CPU2*0J 8*N
fIRG
E__inference_b_norm_4_layer_call_and_return_conditional_losses_27366022"
 b_norm_4/StatefulPartitionedCallш
RELU_4/PartitionedCallPartitionedCall)b_norm_4/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*0
_output_shapes
:         А*-
config_proto

GPU

CPU2*0J 8*L
fGRE
C__inference_RELU_4_layer_call_and_return_conditional_losses_27366312
RELU_4/PartitionedCall№
 max_pooling2d_59/PartitionedCallPartitionedCallRELU_4/PartitionedCall:output:0*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*0
_output_shapes
:         А*-
config_proto

GPU

CPU2*0J 8*V
fQRO
M__inference_max_pooling2d_59_layer_call_and_return_conditional_losses_27361122"
 max_pooling2d_59/PartitionedCallь
flatten_14/PartitionedCallPartitionedCall)max_pooling2d_59/PartitionedCall:output:0*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*(
_output_shapes
:         А*-
config_proto

GPU

CPU2*0J 8*P
fKRI
G__inference_flatten_14_layer_call_and_return_conditional_losses_27366462
flatten_14/PartitionedCallц
dropout_29/PartitionedCallPartitionedCall#flatten_14/PartitionedCall:output:0*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*(
_output_shapes
:         А*-
config_proto

GPU

CPU2*0J 8*P
fKRI
G__inference_dropout_29_layer_call_and_return_conditional_losses_27366792
dropout_29/PartitionedCallб
1output_class_distribution/StatefulPartitionedCallStatefulPartitionedCall#dropout_29/PartitionedCall:output:08output_class_distribution_statefulpartitionedcall_args_18output_class_distribution_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*(
_output_shapes
:         ╚*-
config_proto

GPU

CPU2*0J 8*_
fZRX
V__inference_output_class_distribution_layer_call_and_return_conditional_losses_273670223
1output_class_distribution/StatefulPartitionedCallм
IdentityIdentity:output_class_distribution/StatefulPartitionedCall:output:0!^b_norm_1/StatefulPartitionedCall!^b_norm_2/StatefulPartitionedCall!^b_norm_3/StatefulPartitionedCall!^b_norm_4/StatefulPartitionedCall^conv_1/StatefulPartitionedCall^conv_2/StatefulPartitionedCall^conv_3/StatefulPartitionedCall^conv_4/StatefulPartitionedCall2^input_batch_normalization/StatefulPartitionedCall#^input_conv/StatefulPartitionedCall2^output_class_distribution/StatefulPartitionedCall*
T0*(
_output_shapes
:         ╚2

Identity"
identityIdentity:output:0*▓
_input_shapesа
Э:         АА::::::::::::::::::::::::::::::::2D
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
·
H
,__inference_input_RELU_layer_call_fn_2737698

inputs
identity╝
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*1
_output_shapes
:         АА *-
config_proto

GPU

CPU2*0J 8*P
fKRI
G__inference_input_RELU_layer_call_and_return_conditional_losses_27362092
PartitionedCallv
IdentityIdentityPartitionedCall:output:0*
T0*1
_output_shapes
:         АА 2

Identity"
identityIdentity:output:0*0
_input_shapes
:         АА :& "
 
_user_specified_nameinputs
╥
N
2__inference_max_pooling2d_57_layer_call_fn_2735790

inputs
identity█
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*J
_output_shapes8
6:4                                    *-
config_proto

GPU

CPU2*0J 8*V
fQRO
M__inference_max_pooling2d_57_layer_call_and_return_conditional_losses_27357842
PartitionedCallП
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4                                    2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4                                    :& "
 
_user_specified_nameinputs
А
ш
E__inference_b_norm_3_layer_call_and_return_conditional_losses_2738141

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityИвFusedBatchNormV3/ReadVariableOpв!FusedBatchNormV3/ReadVariableOp_1вReadVariableOpвReadVariableOp_1^
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

LogicalAndu
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:А*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:А*
dtype02
ReadVariableOp_1и
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:А*
dtype02!
FusedBatchNormV3/ReadVariableOpо
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:А*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1с
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,                           А:А:А:А:А:*
epsilon%oГ:*
is_training( 2
FusedBatchNormV3S
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *дp}?2
Constэ
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*B
_output_shapes0
.:,                           А2

Identity"
identityIdentity:output:0*Q
_input_shapes@
>:,                           А::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:& "
 
_user_specified_nameinputs
я
D
(__inference_RELU_4_layer_call_fn_2738413

inputs
identity╖
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*0
_output_shapes
:         А*-
config_proto

GPU

CPU2*0J 8*L
fGRE
C__inference_RELU_4_layer_call_and_return_conditional_losses_27366312
PartitionedCallu
IdentityIdentityPartitionedCall:output:0*
T0*0
_output_shapes
:         А2

Identity"
identityIdentity:output:0*/
_input_shapes
:         А:& "
 
_user_specified_nameinputs
ем
╩
G__inference_awsome_net_layer_call_and_return_conditional_losses_2737454

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
identityИв(b_norm_1/FusedBatchNormV3/ReadVariableOpв*b_norm_1/FusedBatchNormV3/ReadVariableOp_1вb_norm_1/ReadVariableOpвb_norm_1/ReadVariableOp_1в(b_norm_2/FusedBatchNormV3/ReadVariableOpв*b_norm_2/FusedBatchNormV3/ReadVariableOp_1вb_norm_2/ReadVariableOpвb_norm_2/ReadVariableOp_1в(b_norm_3/FusedBatchNormV3/ReadVariableOpв*b_norm_3/FusedBatchNormV3/ReadVariableOp_1вb_norm_3/ReadVariableOpвb_norm_3/ReadVariableOp_1в(b_norm_4/FusedBatchNormV3/ReadVariableOpв*b_norm_4/FusedBatchNormV3/ReadVariableOp_1вb_norm_4/ReadVariableOpвb_norm_4/ReadVariableOp_1вconv_1/BiasAdd/ReadVariableOpвconv_1/Conv2D/ReadVariableOpвconv_2/BiasAdd/ReadVariableOpвconv_2/Conv2D/ReadVariableOpвconv_3/BiasAdd/ReadVariableOpвconv_3/Conv2D/ReadVariableOpвconv_4/BiasAdd/ReadVariableOpвconv_4/Conv2D/ReadVariableOpв9input_batch_normalization/FusedBatchNormV3/ReadVariableOpв;input_batch_normalization/FusedBatchNormV3/ReadVariableOp_1в(input_batch_normalization/ReadVariableOpв*input_batch_normalization/ReadVariableOp_1в!input_conv/BiasAdd/ReadVariableOpв input_conv/Conv2D/ReadVariableOpв0output_class_distribution/BiasAdd/ReadVariableOpв/output_class_distribution/MatMul/ReadVariableOp╢
 input_conv/Conv2D/ReadVariableOpReadVariableOp)input_conv_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02"
 input_conv/Conv2D/ReadVariableOp╞
input_conv/Conv2DConv2Dinputs(input_conv/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:         АА *
paddingSAME*
strides
2
input_conv/Conv2Dн
!input_conv/BiasAdd/ReadVariableOpReadVariableOp*input_conv_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02#
!input_conv/BiasAdd/ReadVariableOp╢
input_conv/BiasAddBiasAddinput_conv/Conv2D:output:0)input_conv/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:         АА 2
input_conv/BiasAddТ
&input_batch_normalization/LogicalAnd/xConst*
_output_shapes
: *
dtype0
*
value	B
 Z 2(
&input_batch_normalization/LogicalAnd/xТ
&input_batch_normalization/LogicalAnd/yConst*
_output_shapes
: *
dtype0
*
value	B
 Z2(
&input_batch_normalization/LogicalAnd/y╘
$input_batch_normalization/LogicalAnd
LogicalAnd/input_batch_normalization/LogicalAnd/x:output:0/input_batch_normalization/LogicalAnd/y:output:0*
_output_shapes
: 2&
$input_batch_normalization/LogicalAnd┬
(input_batch_normalization/ReadVariableOpReadVariableOp1input_batch_normalization_readvariableop_resource*
_output_shapes
: *
dtype02*
(input_batch_normalization/ReadVariableOp╚
*input_batch_normalization/ReadVariableOp_1ReadVariableOp3input_batch_normalization_readvariableop_1_resource*
_output_shapes
: *
dtype02,
*input_batch_normalization/ReadVariableOp_1ї
9input_batch_normalization/FusedBatchNormV3/ReadVariableOpReadVariableOpBinput_batch_normalization_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02;
9input_batch_normalization/FusedBatchNormV3/ReadVariableOp√
;input_batch_normalization/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpDinput_batch_normalization_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02=
;input_batch_normalization/FusedBatchNormV3/ReadVariableOp_1¤
*input_batch_normalization/FusedBatchNormV3FusedBatchNormV3input_conv/BiasAdd:output:00input_batch_normalization/ReadVariableOp:value:02input_batch_normalization/ReadVariableOp_1:value:0Ainput_batch_normalization/FusedBatchNormV3/ReadVariableOp:value:0Cinput_batch_normalization/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*M
_output_shapes;
9:         АА : : : : :*
epsilon%oГ:*
is_training( 2,
*input_batch_normalization/FusedBatchNormV3З
input_batch_normalization/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *дp}?2!
input_batch_normalization/ConstЦ
input_RELU/ReluRelu.input_batch_normalization/FusedBatchNormV3:y:0*
T0*1
_output_shapes
:         АА 2
input_RELU/Relu╦
max_pooling2d_60/MaxPoolMaxPoolinput_RELU/Relu:activations:0*/
_output_shapes
:         @@ *
ksize
*
paddingVALID*
strides
2
max_pooling2d_60/MaxPoolУ
dropout_28/IdentityIdentity!max_pooling2d_60/MaxPool:output:0*
T0*/
_output_shapes
:         @@ 2
dropout_28/Identityк
conv_1/Conv2D/ReadVariableOpReadVariableOp%conv_1_conv2d_readvariableop_resource*&
_output_shapes
:  *
dtype02
conv_1/Conv2D/ReadVariableOp╬
conv_1/Conv2DConv2Ddropout_28/Identity:output:0$conv_1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         @@ *
paddingSAME*
strides
2
conv_1/Conv2Dб
conv_1/BiasAdd/ReadVariableOpReadVariableOp&conv_1_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02
conv_1/BiasAdd/ReadVariableOpд
conv_1/BiasAddBiasAddconv_1/Conv2D:output:0%conv_1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         @@ 2
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
b_norm_1/LogicalAnd/yР
b_norm_1/LogicalAnd
LogicalAndb_norm_1/LogicalAnd/x:output:0b_norm_1/LogicalAnd/y:output:0*
_output_shapes
: 2
b_norm_1/LogicalAndП
b_norm_1/ReadVariableOpReadVariableOp b_norm_1_readvariableop_resource*
_output_shapes
: *
dtype02
b_norm_1/ReadVariableOpХ
b_norm_1/ReadVariableOp_1ReadVariableOp"b_norm_1_readvariableop_1_resource*
_output_shapes
: *
dtype02
b_norm_1/ReadVariableOp_1┬
(b_norm_1/FusedBatchNormV3/ReadVariableOpReadVariableOp1b_norm_1_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02*
(b_norm_1/FusedBatchNormV3/ReadVariableOp╚
*b_norm_1/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp3b_norm_1_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02,
*b_norm_1/FusedBatchNormV3/ReadVariableOp_1С
b_norm_1/FusedBatchNormV3FusedBatchNormV3conv_1/BiasAdd:output:0b_norm_1/ReadVariableOp:value:0!b_norm_1/ReadVariableOp_1:value:00b_norm_1/FusedBatchNormV3/ReadVariableOp:value:02b_norm_1/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:         @@ : : : : :*
epsilon%oГ:*
is_training( 2
b_norm_1/FusedBatchNormV3e
b_norm_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *дp}?2
b_norm_1/Const{
RELU_1/ReluRelub_norm_1/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:         @@ 2
RELU_1/Relu╟
max_pooling2d_56/MaxPoolMaxPoolRELU_1/Relu:activations:0*/
_output_shapes
:            *
ksize
*
paddingVALID*
strides
2
max_pooling2d_56/MaxPoolк
conv_2/Conv2D/ReadVariableOpReadVariableOp%conv_2_conv2d_readvariableop_resource*&
_output_shapes
: @*
dtype02
conv_2/Conv2D/ReadVariableOp╙
conv_2/Conv2DConv2D!max_pooling2d_56/MaxPool:output:0$conv_2/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:           @*
paddingSAME*
strides
2
conv_2/Conv2Dб
conv_2/BiasAdd/ReadVariableOpReadVariableOp&conv_2_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
conv_2/BiasAdd/ReadVariableOpд
conv_2/BiasAddBiasAddconv_2/Conv2D:output:0%conv_2/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:           @2
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
b_norm_2/LogicalAnd/yР
b_norm_2/LogicalAnd
LogicalAndb_norm_2/LogicalAnd/x:output:0b_norm_2/LogicalAnd/y:output:0*
_output_shapes
: 2
b_norm_2/LogicalAndП
b_norm_2/ReadVariableOpReadVariableOp b_norm_2_readvariableop_resource*
_output_shapes
:@*
dtype02
b_norm_2/ReadVariableOpХ
b_norm_2/ReadVariableOp_1ReadVariableOp"b_norm_2_readvariableop_1_resource*
_output_shapes
:@*
dtype02
b_norm_2/ReadVariableOp_1┬
(b_norm_2/FusedBatchNormV3/ReadVariableOpReadVariableOp1b_norm_2_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02*
(b_norm_2/FusedBatchNormV3/ReadVariableOp╚
*b_norm_2/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp3b_norm_2_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02,
*b_norm_2/FusedBatchNormV3/ReadVariableOp_1С
b_norm_2/FusedBatchNormV3FusedBatchNormV3conv_2/BiasAdd:output:0b_norm_2/ReadVariableOp:value:0!b_norm_2/ReadVariableOp_1:value:00b_norm_2/FusedBatchNormV3/ReadVariableOp:value:02b_norm_2/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:           @:@:@:@:@:*
epsilon%oГ:*
is_training( 2
b_norm_2/FusedBatchNormV3e
b_norm_2/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *дp}?2
b_norm_2/Const{
RELU_2/ReluRelub_norm_2/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:           @2
RELU_2/Relu╟
max_pooling2d_57/MaxPoolMaxPoolRELU_2/Relu:activations:0*/
_output_shapes
:         @*
ksize
*
paddingVALID*
strides
2
max_pooling2d_57/MaxPoolл
conv_3/Conv2D/ReadVariableOpReadVariableOp%conv_3_conv2d_readvariableop_resource*'
_output_shapes
:@А*
dtype02
conv_3/Conv2D/ReadVariableOp╘
conv_3/Conv2DConv2D!max_pooling2d_57/MaxPool:output:0$conv_3/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:         А*
paddingSAME*
strides
2
conv_3/Conv2Dв
conv_3/BiasAdd/ReadVariableOpReadVariableOp&conv_3_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02
conv_3/BiasAdd/ReadVariableOpе
conv_3/BiasAddBiasAddconv_3/Conv2D:output:0%conv_3/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:         А2
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
b_norm_3/LogicalAnd/yР
b_norm_3/LogicalAnd
LogicalAndb_norm_3/LogicalAnd/x:output:0b_norm_3/LogicalAnd/y:output:0*
_output_shapes
: 2
b_norm_3/LogicalAndР
b_norm_3/ReadVariableOpReadVariableOp b_norm_3_readvariableop_resource*
_output_shapes	
:А*
dtype02
b_norm_3/ReadVariableOpЦ
b_norm_3/ReadVariableOp_1ReadVariableOp"b_norm_3_readvariableop_1_resource*
_output_shapes	
:А*
dtype02
b_norm_3/ReadVariableOp_1├
(b_norm_3/FusedBatchNormV3/ReadVariableOpReadVariableOp1b_norm_3_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:А*
dtype02*
(b_norm_3/FusedBatchNormV3/ReadVariableOp╔
*b_norm_3/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp3b_norm_3_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:А*
dtype02,
*b_norm_3/FusedBatchNormV3/ReadVariableOp_1Ц
b_norm_3/FusedBatchNormV3FusedBatchNormV3conv_3/BiasAdd:output:0b_norm_3/ReadVariableOp:value:0!b_norm_3/ReadVariableOp_1:value:00b_norm_3/FusedBatchNormV3/ReadVariableOp:value:02b_norm_3/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:         А:А:А:А:А:*
epsilon%oГ:*
is_training( 2
b_norm_3/FusedBatchNormV3e
b_norm_3/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *дp}?2
b_norm_3/Const|
RELU_3/ReluRelub_norm_3/FusedBatchNormV3:y:0*
T0*0
_output_shapes
:         А2
RELU_3/Relu╚
max_pooling2d_58/MaxPoolMaxPoolRELU_3/Relu:activations:0*0
_output_shapes
:         А*
ksize
*
paddingVALID*
strides
2
max_pooling2d_58/MaxPoolм
conv_4/Conv2D/ReadVariableOpReadVariableOp%conv_4_conv2d_readvariableop_resource*(
_output_shapes
:АА*
dtype02
conv_4/Conv2D/ReadVariableOp╘
conv_4/Conv2DConv2D!max_pooling2d_58/MaxPool:output:0$conv_4/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:         А*
paddingSAME*
strides
2
conv_4/Conv2Dв
conv_4/BiasAdd/ReadVariableOpReadVariableOp&conv_4_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02
conv_4/BiasAdd/ReadVariableOpе
conv_4/BiasAddBiasAddconv_4/Conv2D:output:0%conv_4/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:         А2
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
b_norm_4/LogicalAnd/yР
b_norm_4/LogicalAnd
LogicalAndb_norm_4/LogicalAnd/x:output:0b_norm_4/LogicalAnd/y:output:0*
_output_shapes
: 2
b_norm_4/LogicalAndР
b_norm_4/ReadVariableOpReadVariableOp b_norm_4_readvariableop_resource*
_output_shapes	
:А*
dtype02
b_norm_4/ReadVariableOpЦ
b_norm_4/ReadVariableOp_1ReadVariableOp"b_norm_4_readvariableop_1_resource*
_output_shapes	
:А*
dtype02
b_norm_4/ReadVariableOp_1├
(b_norm_4/FusedBatchNormV3/ReadVariableOpReadVariableOp1b_norm_4_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:А*
dtype02*
(b_norm_4/FusedBatchNormV3/ReadVariableOp╔
*b_norm_4/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp3b_norm_4_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:А*
dtype02,
*b_norm_4/FusedBatchNormV3/ReadVariableOp_1Ц
b_norm_4/FusedBatchNormV3FusedBatchNormV3conv_4/BiasAdd:output:0b_norm_4/ReadVariableOp:value:0!b_norm_4/ReadVariableOp_1:value:00b_norm_4/FusedBatchNormV3/ReadVariableOp:value:02b_norm_4/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:         А:А:А:А:А:*
epsilon%oГ:*
is_training( 2
b_norm_4/FusedBatchNormV3e
b_norm_4/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *дp}?2
b_norm_4/Const|
RELU_4/ReluRelub_norm_4/FusedBatchNormV3:y:0*
T0*0
_output_shapes
:         А2
RELU_4/Relu╚
max_pooling2d_59/MaxPoolMaxPoolRELU_4/Relu:activations:0*0
_output_shapes
:         А*
ksize
*
paddingVALID*
strides
2
max_pooling2d_59/MaxPoolu
flatten_14/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2
flatten_14/Constд
flatten_14/ReshapeReshape!max_pooling2d_59/MaxPool:output:0flatten_14/Const:output:0*
T0*(
_output_shapes
:         А2
flatten_14/ReshapeЖ
dropout_29/IdentityIdentityflatten_14/Reshape:output:0*
T0*(
_output_shapes
:         А2
dropout_29/Identity▌
/output_class_distribution/MatMul/ReadVariableOpReadVariableOp8output_class_distribution_matmul_readvariableop_resource* 
_output_shapes
:
А╚*
dtype021
/output_class_distribution/MatMul/ReadVariableOp╪
 output_class_distribution/MatMulMatMuldropout_29/Identity:output:07output_class_distribution/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ╚2"
 output_class_distribution/MatMul█
0output_class_distribution/BiasAdd/ReadVariableOpReadVariableOp9output_class_distribution_biasadd_readvariableop_resource*
_output_shapes	
:╚*
dtype022
0output_class_distribution/BiasAdd/ReadVariableOpъ
!output_class_distribution/BiasAddBiasAdd*output_class_distribution/MatMul:product:08output_class_distribution/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ╚2#
!output_class_distribution/BiasAdd▒

IdentityIdentity*output_class_distribution/BiasAdd:output:0)^b_norm_1/FusedBatchNormV3/ReadVariableOp+^b_norm_1/FusedBatchNormV3/ReadVariableOp_1^b_norm_1/ReadVariableOp^b_norm_1/ReadVariableOp_1)^b_norm_2/FusedBatchNormV3/ReadVariableOp+^b_norm_2/FusedBatchNormV3/ReadVariableOp_1^b_norm_2/ReadVariableOp^b_norm_2/ReadVariableOp_1)^b_norm_3/FusedBatchNormV3/ReadVariableOp+^b_norm_3/FusedBatchNormV3/ReadVariableOp_1^b_norm_3/ReadVariableOp^b_norm_3/ReadVariableOp_1)^b_norm_4/FusedBatchNormV3/ReadVariableOp+^b_norm_4/FusedBatchNormV3/ReadVariableOp_1^b_norm_4/ReadVariableOp^b_norm_4/ReadVariableOp_1^conv_1/BiasAdd/ReadVariableOp^conv_1/Conv2D/ReadVariableOp^conv_2/BiasAdd/ReadVariableOp^conv_2/Conv2D/ReadVariableOp^conv_3/BiasAdd/ReadVariableOp^conv_3/Conv2D/ReadVariableOp^conv_4/BiasAdd/ReadVariableOp^conv_4/Conv2D/ReadVariableOp:^input_batch_normalization/FusedBatchNormV3/ReadVariableOp<^input_batch_normalization/FusedBatchNormV3/ReadVariableOp_1)^input_batch_normalization/ReadVariableOp+^input_batch_normalization/ReadVariableOp_1"^input_conv/BiasAdd/ReadVariableOp!^input_conv/Conv2D/ReadVariableOp1^output_class_distribution/BiasAdd/ReadVariableOp0^output_class_distribution/MatMul/ReadVariableOp*
T0*(
_output_shapes
:         ╚2

Identity"
identityIdentity:output:0*▓
_input_shapesа
Э:         АА::::::::::::::::::::::::::::::::2T
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
б
є
*__inference_b_norm_2_layer_call_fn_2738054

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4
identityИвStatefulPartitionedCall╘
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4*
Tin	
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*/
_output_shapes
:           @*-
config_proto

GPU

CPU2*0J 8*N
fIRG
E__inference_b_norm_2_layer_call_and_return_conditional_losses_27363882
StatefulPartitionedCallЦ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:           @2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:           @::::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs
╩
ш
E__inference_b_norm_4_layer_call_and_return_conditional_losses_2736602

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityИвFusedBatchNormV3/ReadVariableOpв!FusedBatchNormV3/ReadVariableOp_1вReadVariableOpвReadVariableOp_1^
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

LogicalAndu
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:А*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:А*
dtype02
ReadVariableOp_1и
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:А*
dtype02!
FusedBatchNormV3/ReadVariableOpо
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:А*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1╧
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:         А:А:А:А:А:*
epsilon%oГ:*
is_training( 2
FusedBatchNormV3S
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *дp}?2
Const█
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*0
_output_shapes
:         А2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:         А::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:& "
 
_user_specified_nameinputs
б
є
*__inference_b_norm_2_layer_call_fn_2738063

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4
identityИвStatefulPartitionedCall╘
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4*
Tin	
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*/
_output_shapes
:           @*-
config_proto

GPU

CPU2*0J 8*N
fIRG
E__inference_b_norm_2_layer_call_and_return_conditional_losses_27364102
StatefulPartitionedCallЦ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:           @2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:           @::::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs
▓╤
я
"__inference__wrapped_model_2735298
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
identityИв3awsome_net/b_norm_1/FusedBatchNormV3/ReadVariableOpв5awsome_net/b_norm_1/FusedBatchNormV3/ReadVariableOp_1в"awsome_net/b_norm_1/ReadVariableOpв$awsome_net/b_norm_1/ReadVariableOp_1в3awsome_net/b_norm_2/FusedBatchNormV3/ReadVariableOpв5awsome_net/b_norm_2/FusedBatchNormV3/ReadVariableOp_1в"awsome_net/b_norm_2/ReadVariableOpв$awsome_net/b_norm_2/ReadVariableOp_1в3awsome_net/b_norm_3/FusedBatchNormV3/ReadVariableOpв5awsome_net/b_norm_3/FusedBatchNormV3/ReadVariableOp_1в"awsome_net/b_norm_3/ReadVariableOpв$awsome_net/b_norm_3/ReadVariableOp_1в3awsome_net/b_norm_4/FusedBatchNormV3/ReadVariableOpв5awsome_net/b_norm_4/FusedBatchNormV3/ReadVariableOp_1в"awsome_net/b_norm_4/ReadVariableOpв$awsome_net/b_norm_4/ReadVariableOp_1в(awsome_net/conv_1/BiasAdd/ReadVariableOpв'awsome_net/conv_1/Conv2D/ReadVariableOpв(awsome_net/conv_2/BiasAdd/ReadVariableOpв'awsome_net/conv_2/Conv2D/ReadVariableOpв(awsome_net/conv_3/BiasAdd/ReadVariableOpв'awsome_net/conv_3/Conv2D/ReadVariableOpв(awsome_net/conv_4/BiasAdd/ReadVariableOpв'awsome_net/conv_4/Conv2D/ReadVariableOpвDawsome_net/input_batch_normalization/FusedBatchNormV3/ReadVariableOpвFawsome_net/input_batch_normalization/FusedBatchNormV3/ReadVariableOp_1в3awsome_net/input_batch_normalization/ReadVariableOpв5awsome_net/input_batch_normalization/ReadVariableOp_1в,awsome_net/input_conv/BiasAdd/ReadVariableOpв+awsome_net/input_conv/Conv2D/ReadVariableOpв;awsome_net/output_class_distribution/BiasAdd/ReadVariableOpв:awsome_net/output_class_distribution/MatMul/ReadVariableOp╫
+awsome_net/input_conv/Conv2D/ReadVariableOpReadVariableOp4awsome_net_input_conv_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02-
+awsome_net/input_conv/Conv2D/ReadVariableOpё
awsome_net/input_conv/Conv2DConv2Dinput_conv_input3awsome_net/input_conv/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:         АА *
paddingSAME*
strides
2
awsome_net/input_conv/Conv2D╬
,awsome_net/input_conv/BiasAdd/ReadVariableOpReadVariableOp5awsome_net_input_conv_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02.
,awsome_net/input_conv/BiasAdd/ReadVariableOpт
awsome_net/input_conv/BiasAddBiasAdd%awsome_net/input_conv/Conv2D:output:04awsome_net/input_conv/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:         АА 2
awsome_net/input_conv/BiasAddи
1awsome_net/input_batch_normalization/LogicalAnd/xConst*
_output_shapes
: *
dtype0
*
value	B
 Z 23
1awsome_net/input_batch_normalization/LogicalAnd/xи
1awsome_net/input_batch_normalization/LogicalAnd/yConst*
_output_shapes
: *
dtype0
*
value	B
 Z23
1awsome_net/input_batch_normalization/LogicalAnd/yА
/awsome_net/input_batch_normalization/LogicalAnd
LogicalAnd:awsome_net/input_batch_normalization/LogicalAnd/x:output:0:awsome_net/input_batch_normalization/LogicalAnd/y:output:0*
_output_shapes
: 21
/awsome_net/input_batch_normalization/LogicalAndу
3awsome_net/input_batch_normalization/ReadVariableOpReadVariableOp<awsome_net_input_batch_normalization_readvariableop_resource*
_output_shapes
: *
dtype025
3awsome_net/input_batch_normalization/ReadVariableOpщ
5awsome_net/input_batch_normalization/ReadVariableOp_1ReadVariableOp>awsome_net_input_batch_normalization_readvariableop_1_resource*
_output_shapes
: *
dtype027
5awsome_net/input_batch_normalization/ReadVariableOp_1Ц
Dawsome_net/input_batch_normalization/FusedBatchNormV3/ReadVariableOpReadVariableOpMawsome_net_input_batch_normalization_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02F
Dawsome_net/input_batch_normalization/FusedBatchNormV3/ReadVariableOpЬ
Fawsome_net/input_batch_normalization/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpOawsome_net_input_batch_normalization_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02H
Fawsome_net/input_batch_normalization/FusedBatchNormV3/ReadVariableOp_1╩
5awsome_net/input_batch_normalization/FusedBatchNormV3FusedBatchNormV3&awsome_net/input_conv/BiasAdd:output:0;awsome_net/input_batch_normalization/ReadVariableOp:value:0=awsome_net/input_batch_normalization/ReadVariableOp_1:value:0Lawsome_net/input_batch_normalization/FusedBatchNormV3/ReadVariableOp:value:0Nawsome_net/input_batch_normalization/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*M
_output_shapes;
9:         АА : : : : :*
epsilon%oГ:*
is_training( 27
5awsome_net/input_batch_normalization/FusedBatchNormV3Э
*awsome_net/input_batch_normalization/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *дp}?2,
*awsome_net/input_batch_normalization/Const╖
awsome_net/input_RELU/ReluRelu9awsome_net/input_batch_normalization/FusedBatchNormV3:y:0*
T0*1
_output_shapes
:         АА 2
awsome_net/input_RELU/Reluь
#awsome_net/max_pooling2d_60/MaxPoolMaxPool(awsome_net/input_RELU/Relu:activations:0*/
_output_shapes
:         @@ *
ksize
*
paddingVALID*
strides
2%
#awsome_net/max_pooling2d_60/MaxPool┤
awsome_net/dropout_28/IdentityIdentity,awsome_net/max_pooling2d_60/MaxPool:output:0*
T0*/
_output_shapes
:         @@ 2 
awsome_net/dropout_28/Identity╦
'awsome_net/conv_1/Conv2D/ReadVariableOpReadVariableOp0awsome_net_conv_1_conv2d_readvariableop_resource*&
_output_shapes
:  *
dtype02)
'awsome_net/conv_1/Conv2D/ReadVariableOp·
awsome_net/conv_1/Conv2DConv2D'awsome_net/dropout_28/Identity:output:0/awsome_net/conv_1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         @@ *
paddingSAME*
strides
2
awsome_net/conv_1/Conv2D┬
(awsome_net/conv_1/BiasAdd/ReadVariableOpReadVariableOp1awsome_net_conv_1_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02*
(awsome_net/conv_1/BiasAdd/ReadVariableOp╨
awsome_net/conv_1/BiasAddBiasAdd!awsome_net/conv_1/Conv2D:output:00awsome_net/conv_1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         @@ 2
awsome_net/conv_1/BiasAddЖ
 awsome_net/b_norm_1/LogicalAnd/xConst*
_output_shapes
: *
dtype0
*
value	B
 Z 2"
 awsome_net/b_norm_1/LogicalAnd/xЖ
 awsome_net/b_norm_1/LogicalAnd/yConst*
_output_shapes
: *
dtype0
*
value	B
 Z2"
 awsome_net/b_norm_1/LogicalAnd/y╝
awsome_net/b_norm_1/LogicalAnd
LogicalAnd)awsome_net/b_norm_1/LogicalAnd/x:output:0)awsome_net/b_norm_1/LogicalAnd/y:output:0*
_output_shapes
: 2 
awsome_net/b_norm_1/LogicalAnd░
"awsome_net/b_norm_1/ReadVariableOpReadVariableOp+awsome_net_b_norm_1_readvariableop_resource*
_output_shapes
: *
dtype02$
"awsome_net/b_norm_1/ReadVariableOp╢
$awsome_net/b_norm_1/ReadVariableOp_1ReadVariableOp-awsome_net_b_norm_1_readvariableop_1_resource*
_output_shapes
: *
dtype02&
$awsome_net/b_norm_1/ReadVariableOp_1у
3awsome_net/b_norm_1/FusedBatchNormV3/ReadVariableOpReadVariableOp<awsome_net_b_norm_1_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype025
3awsome_net/b_norm_1/FusedBatchNormV3/ReadVariableOpщ
5awsome_net/b_norm_1/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp>awsome_net_b_norm_1_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype027
5awsome_net/b_norm_1/FusedBatchNormV3/ReadVariableOp_1▐
$awsome_net/b_norm_1/FusedBatchNormV3FusedBatchNormV3"awsome_net/conv_1/BiasAdd:output:0*awsome_net/b_norm_1/ReadVariableOp:value:0,awsome_net/b_norm_1/ReadVariableOp_1:value:0;awsome_net/b_norm_1/FusedBatchNormV3/ReadVariableOp:value:0=awsome_net/b_norm_1/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:         @@ : : : : :*
epsilon%oГ:*
is_training( 2&
$awsome_net/b_norm_1/FusedBatchNormV3{
awsome_net/b_norm_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *дp}?2
awsome_net/b_norm_1/ConstЬ
awsome_net/RELU_1/ReluRelu(awsome_net/b_norm_1/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:         @@ 2
awsome_net/RELU_1/Reluш
#awsome_net/max_pooling2d_56/MaxPoolMaxPool$awsome_net/RELU_1/Relu:activations:0*/
_output_shapes
:            *
ksize
*
paddingVALID*
strides
2%
#awsome_net/max_pooling2d_56/MaxPool╦
'awsome_net/conv_2/Conv2D/ReadVariableOpReadVariableOp0awsome_net_conv_2_conv2d_readvariableop_resource*&
_output_shapes
: @*
dtype02)
'awsome_net/conv_2/Conv2D/ReadVariableOp 
awsome_net/conv_2/Conv2DConv2D,awsome_net/max_pooling2d_56/MaxPool:output:0/awsome_net/conv_2/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:           @*
paddingSAME*
strides
2
awsome_net/conv_2/Conv2D┬
(awsome_net/conv_2/BiasAdd/ReadVariableOpReadVariableOp1awsome_net_conv_2_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02*
(awsome_net/conv_2/BiasAdd/ReadVariableOp╨
awsome_net/conv_2/BiasAddBiasAdd!awsome_net/conv_2/Conv2D:output:00awsome_net/conv_2/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:           @2
awsome_net/conv_2/BiasAddЖ
 awsome_net/b_norm_2/LogicalAnd/xConst*
_output_shapes
: *
dtype0
*
value	B
 Z 2"
 awsome_net/b_norm_2/LogicalAnd/xЖ
 awsome_net/b_norm_2/LogicalAnd/yConst*
_output_shapes
: *
dtype0
*
value	B
 Z2"
 awsome_net/b_norm_2/LogicalAnd/y╝
awsome_net/b_norm_2/LogicalAnd
LogicalAnd)awsome_net/b_norm_2/LogicalAnd/x:output:0)awsome_net/b_norm_2/LogicalAnd/y:output:0*
_output_shapes
: 2 
awsome_net/b_norm_2/LogicalAnd░
"awsome_net/b_norm_2/ReadVariableOpReadVariableOp+awsome_net_b_norm_2_readvariableop_resource*
_output_shapes
:@*
dtype02$
"awsome_net/b_norm_2/ReadVariableOp╢
$awsome_net/b_norm_2/ReadVariableOp_1ReadVariableOp-awsome_net_b_norm_2_readvariableop_1_resource*
_output_shapes
:@*
dtype02&
$awsome_net/b_norm_2/ReadVariableOp_1у
3awsome_net/b_norm_2/FusedBatchNormV3/ReadVariableOpReadVariableOp<awsome_net_b_norm_2_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype025
3awsome_net/b_norm_2/FusedBatchNormV3/ReadVariableOpщ
5awsome_net/b_norm_2/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp>awsome_net_b_norm_2_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype027
5awsome_net/b_norm_2/FusedBatchNormV3/ReadVariableOp_1▐
$awsome_net/b_norm_2/FusedBatchNormV3FusedBatchNormV3"awsome_net/conv_2/BiasAdd:output:0*awsome_net/b_norm_2/ReadVariableOp:value:0,awsome_net/b_norm_2/ReadVariableOp_1:value:0;awsome_net/b_norm_2/FusedBatchNormV3/ReadVariableOp:value:0=awsome_net/b_norm_2/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:           @:@:@:@:@:*
epsilon%oГ:*
is_training( 2&
$awsome_net/b_norm_2/FusedBatchNormV3{
awsome_net/b_norm_2/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *дp}?2
awsome_net/b_norm_2/ConstЬ
awsome_net/RELU_2/ReluRelu(awsome_net/b_norm_2/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:           @2
awsome_net/RELU_2/Reluш
#awsome_net/max_pooling2d_57/MaxPoolMaxPool$awsome_net/RELU_2/Relu:activations:0*/
_output_shapes
:         @*
ksize
*
paddingVALID*
strides
2%
#awsome_net/max_pooling2d_57/MaxPool╠
'awsome_net/conv_3/Conv2D/ReadVariableOpReadVariableOp0awsome_net_conv_3_conv2d_readvariableop_resource*'
_output_shapes
:@А*
dtype02)
'awsome_net/conv_3/Conv2D/ReadVariableOpА
awsome_net/conv_3/Conv2DConv2D,awsome_net/max_pooling2d_57/MaxPool:output:0/awsome_net/conv_3/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:         А*
paddingSAME*
strides
2
awsome_net/conv_3/Conv2D├
(awsome_net/conv_3/BiasAdd/ReadVariableOpReadVariableOp1awsome_net_conv_3_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02*
(awsome_net/conv_3/BiasAdd/ReadVariableOp╤
awsome_net/conv_3/BiasAddBiasAdd!awsome_net/conv_3/Conv2D:output:00awsome_net/conv_3/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:         А2
awsome_net/conv_3/BiasAddЖ
 awsome_net/b_norm_3/LogicalAnd/xConst*
_output_shapes
: *
dtype0
*
value	B
 Z 2"
 awsome_net/b_norm_3/LogicalAnd/xЖ
 awsome_net/b_norm_3/LogicalAnd/yConst*
_output_shapes
: *
dtype0
*
value	B
 Z2"
 awsome_net/b_norm_3/LogicalAnd/y╝
awsome_net/b_norm_3/LogicalAnd
LogicalAnd)awsome_net/b_norm_3/LogicalAnd/x:output:0)awsome_net/b_norm_3/LogicalAnd/y:output:0*
_output_shapes
: 2 
awsome_net/b_norm_3/LogicalAnd▒
"awsome_net/b_norm_3/ReadVariableOpReadVariableOp+awsome_net_b_norm_3_readvariableop_resource*
_output_shapes	
:А*
dtype02$
"awsome_net/b_norm_3/ReadVariableOp╖
$awsome_net/b_norm_3/ReadVariableOp_1ReadVariableOp-awsome_net_b_norm_3_readvariableop_1_resource*
_output_shapes	
:А*
dtype02&
$awsome_net/b_norm_3/ReadVariableOp_1ф
3awsome_net/b_norm_3/FusedBatchNormV3/ReadVariableOpReadVariableOp<awsome_net_b_norm_3_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:А*
dtype025
3awsome_net/b_norm_3/FusedBatchNormV3/ReadVariableOpъ
5awsome_net/b_norm_3/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp>awsome_net_b_norm_3_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:А*
dtype027
5awsome_net/b_norm_3/FusedBatchNormV3/ReadVariableOp_1у
$awsome_net/b_norm_3/FusedBatchNormV3FusedBatchNormV3"awsome_net/conv_3/BiasAdd:output:0*awsome_net/b_norm_3/ReadVariableOp:value:0,awsome_net/b_norm_3/ReadVariableOp_1:value:0;awsome_net/b_norm_3/FusedBatchNormV3/ReadVariableOp:value:0=awsome_net/b_norm_3/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:         А:А:А:А:А:*
epsilon%oГ:*
is_training( 2&
$awsome_net/b_norm_3/FusedBatchNormV3{
awsome_net/b_norm_3/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *дp}?2
awsome_net/b_norm_3/ConstЭ
awsome_net/RELU_3/ReluRelu(awsome_net/b_norm_3/FusedBatchNormV3:y:0*
T0*0
_output_shapes
:         А2
awsome_net/RELU_3/Reluщ
#awsome_net/max_pooling2d_58/MaxPoolMaxPool$awsome_net/RELU_3/Relu:activations:0*0
_output_shapes
:         А*
ksize
*
paddingVALID*
strides
2%
#awsome_net/max_pooling2d_58/MaxPool═
'awsome_net/conv_4/Conv2D/ReadVariableOpReadVariableOp0awsome_net_conv_4_conv2d_readvariableop_resource*(
_output_shapes
:АА*
dtype02)
'awsome_net/conv_4/Conv2D/ReadVariableOpА
awsome_net/conv_4/Conv2DConv2D,awsome_net/max_pooling2d_58/MaxPool:output:0/awsome_net/conv_4/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:         А*
paddingSAME*
strides
2
awsome_net/conv_4/Conv2D├
(awsome_net/conv_4/BiasAdd/ReadVariableOpReadVariableOp1awsome_net_conv_4_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02*
(awsome_net/conv_4/BiasAdd/ReadVariableOp╤
awsome_net/conv_4/BiasAddBiasAdd!awsome_net/conv_4/Conv2D:output:00awsome_net/conv_4/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:         А2
awsome_net/conv_4/BiasAddЖ
 awsome_net/b_norm_4/LogicalAnd/xConst*
_output_shapes
: *
dtype0
*
value	B
 Z 2"
 awsome_net/b_norm_4/LogicalAnd/xЖ
 awsome_net/b_norm_4/LogicalAnd/yConst*
_output_shapes
: *
dtype0
*
value	B
 Z2"
 awsome_net/b_norm_4/LogicalAnd/y╝
awsome_net/b_norm_4/LogicalAnd
LogicalAnd)awsome_net/b_norm_4/LogicalAnd/x:output:0)awsome_net/b_norm_4/LogicalAnd/y:output:0*
_output_shapes
: 2 
awsome_net/b_norm_4/LogicalAnd▒
"awsome_net/b_norm_4/ReadVariableOpReadVariableOp+awsome_net_b_norm_4_readvariableop_resource*
_output_shapes	
:А*
dtype02$
"awsome_net/b_norm_4/ReadVariableOp╖
$awsome_net/b_norm_4/ReadVariableOp_1ReadVariableOp-awsome_net_b_norm_4_readvariableop_1_resource*
_output_shapes	
:А*
dtype02&
$awsome_net/b_norm_4/ReadVariableOp_1ф
3awsome_net/b_norm_4/FusedBatchNormV3/ReadVariableOpReadVariableOp<awsome_net_b_norm_4_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:А*
dtype025
3awsome_net/b_norm_4/FusedBatchNormV3/ReadVariableOpъ
5awsome_net/b_norm_4/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp>awsome_net_b_norm_4_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:А*
dtype027
5awsome_net/b_norm_4/FusedBatchNormV3/ReadVariableOp_1у
$awsome_net/b_norm_4/FusedBatchNormV3FusedBatchNormV3"awsome_net/conv_4/BiasAdd:output:0*awsome_net/b_norm_4/ReadVariableOp:value:0,awsome_net/b_norm_4/ReadVariableOp_1:value:0;awsome_net/b_norm_4/FusedBatchNormV3/ReadVariableOp:value:0=awsome_net/b_norm_4/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:         А:А:А:А:А:*
epsilon%oГ:*
is_training( 2&
$awsome_net/b_norm_4/FusedBatchNormV3{
awsome_net/b_norm_4/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *дp}?2
awsome_net/b_norm_4/ConstЭ
awsome_net/RELU_4/ReluRelu(awsome_net/b_norm_4/FusedBatchNormV3:y:0*
T0*0
_output_shapes
:         А2
awsome_net/RELU_4/Reluщ
#awsome_net/max_pooling2d_59/MaxPoolMaxPool$awsome_net/RELU_4/Relu:activations:0*0
_output_shapes
:         А*
ksize
*
paddingVALID*
strides
2%
#awsome_net/max_pooling2d_59/MaxPoolЛ
awsome_net/flatten_14/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2
awsome_net/flatten_14/Const╨
awsome_net/flatten_14/ReshapeReshape,awsome_net/max_pooling2d_59/MaxPool:output:0$awsome_net/flatten_14/Const:output:0*
T0*(
_output_shapes
:         А2
awsome_net/flatten_14/Reshapeз
awsome_net/dropout_29/IdentityIdentity&awsome_net/flatten_14/Reshape:output:0*
T0*(
_output_shapes
:         А2 
awsome_net/dropout_29/Identity■
:awsome_net/output_class_distribution/MatMul/ReadVariableOpReadVariableOpCawsome_net_output_class_distribution_matmul_readvariableop_resource* 
_output_shapes
:
А╚*
dtype02<
:awsome_net/output_class_distribution/MatMul/ReadVariableOpД
+awsome_net/output_class_distribution/MatMulMatMul'awsome_net/dropout_29/Identity:output:0Bawsome_net/output_class_distribution/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ╚2-
+awsome_net/output_class_distribution/MatMul№
;awsome_net/output_class_distribution/BiasAdd/ReadVariableOpReadVariableOpDawsome_net_output_class_distribution_biasadd_readvariableop_resource*
_output_shapes	
:╚*
dtype02=
;awsome_net/output_class_distribution/BiasAdd/ReadVariableOpЦ
,awsome_net/output_class_distribution/BiasAddBiasAdd5awsome_net/output_class_distribution/MatMul:product:0Cawsome_net/output_class_distribution/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ╚2.
,awsome_net/output_class_distribution/BiasAddЬ
IdentityIdentity5awsome_net/output_class_distribution/BiasAdd:output:04^awsome_net/b_norm_1/FusedBatchNormV3/ReadVariableOp6^awsome_net/b_norm_1/FusedBatchNormV3/ReadVariableOp_1#^awsome_net/b_norm_1/ReadVariableOp%^awsome_net/b_norm_1/ReadVariableOp_14^awsome_net/b_norm_2/FusedBatchNormV3/ReadVariableOp6^awsome_net/b_norm_2/FusedBatchNormV3/ReadVariableOp_1#^awsome_net/b_norm_2/ReadVariableOp%^awsome_net/b_norm_2/ReadVariableOp_14^awsome_net/b_norm_3/FusedBatchNormV3/ReadVariableOp6^awsome_net/b_norm_3/FusedBatchNormV3/ReadVariableOp_1#^awsome_net/b_norm_3/ReadVariableOp%^awsome_net/b_norm_3/ReadVariableOp_14^awsome_net/b_norm_4/FusedBatchNormV3/ReadVariableOp6^awsome_net/b_norm_4/FusedBatchNormV3/ReadVariableOp_1#^awsome_net/b_norm_4/ReadVariableOp%^awsome_net/b_norm_4/ReadVariableOp_1)^awsome_net/conv_1/BiasAdd/ReadVariableOp(^awsome_net/conv_1/Conv2D/ReadVariableOp)^awsome_net/conv_2/BiasAdd/ReadVariableOp(^awsome_net/conv_2/Conv2D/ReadVariableOp)^awsome_net/conv_3/BiasAdd/ReadVariableOp(^awsome_net/conv_3/Conv2D/ReadVariableOp)^awsome_net/conv_4/BiasAdd/ReadVariableOp(^awsome_net/conv_4/Conv2D/ReadVariableOpE^awsome_net/input_batch_normalization/FusedBatchNormV3/ReadVariableOpG^awsome_net/input_batch_normalization/FusedBatchNormV3/ReadVariableOp_14^awsome_net/input_batch_normalization/ReadVariableOp6^awsome_net/input_batch_normalization/ReadVariableOp_1-^awsome_net/input_conv/BiasAdd/ReadVariableOp,^awsome_net/input_conv/Conv2D/ReadVariableOp<^awsome_net/output_class_distribution/BiasAdd/ReadVariableOp;^awsome_net/output_class_distribution/MatMul/ReadVariableOp*
T0*(
_output_shapes
:         ╚2

Identity"
identityIdentity:output:0*▓
_input_shapesа
Э:         АА::::::::::::::::::::::::::::::::2j
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
'awsome_net/conv_4/Conv2D/ReadVariableOp'awsome_net/conv_4/Conv2D/ReadVariableOp2М
Dawsome_net/input_batch_normalization/FusedBatchNormV3/ReadVariableOpDawsome_net/input_batch_normalization/FusedBatchNormV3/ReadVariableOp2Р
Fawsome_net/input_batch_normalization/FusedBatchNormV3/ReadVariableOp_1Fawsome_net/input_batch_normalization/FusedBatchNormV3/ReadVariableOp_12j
3awsome_net/input_batch_normalization/ReadVariableOp3awsome_net/input_batch_normalization/ReadVariableOp2n
5awsome_net/input_batch_normalization/ReadVariableOp_15awsome_net/input_batch_normalization/ReadVariableOp_12\
,awsome_net/input_conv/BiasAdd/ReadVariableOp,awsome_net/input_conv/BiasAdd/ReadVariableOp2Z
+awsome_net/input_conv/Conv2D/ReadVariableOp+awsome_net/input_conv/Conv2D/ReadVariableOp2z
;awsome_net/output_class_distribution/BiasAdd/ReadVariableOp;awsome_net/output_class_distribution/BiasAdd/ReadVariableOp2x
:awsome_net/output_class_distribution/MatMul/ReadVariableOp:awsome_net/output_class_distribution/MatMul/ReadVariableOp:0 ,
*
_user_specified_nameinput_conv_input
╔
н
,__inference_input_conv_layer_call_fn_2735318

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2
identityИвStatefulPartitionedCallж
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*A
_output_shapes/
-:+                            *-
config_proto

GPU

CPU2*0J 8*P
fKRI
G__inference_input_conv_layer_call_and_return_conditional_losses_27353102
StatefulPartitionedCallи
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+                            2

Identity"
identityIdentity:output:0*H
_input_shapes7
5:+                           ::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs
Я$
г
V__inference_input_batch_normalization_layer_call_and_return_conditional_losses_2736158

inputs
readvariableop_resource
readvariableop_1_resource
assignmovingavg_2736143
assignmovingavg_1_2736150
identityИв#AssignMovingAvg/AssignSubVariableOpвAssignMovingAvg/ReadVariableOpв%AssignMovingAvg_1/AssignSubVariableOpв AssignMovingAvg_1/ReadVariableOpвReadVariableOpвReadVariableOp_1^
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
Const_1З
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0Const:output:0Const_1:output:0*
T0*
U0*M
_output_shapes;
9:         АА : : : : :*
epsilon%oГ:2
FusedBatchNormV3W
Const_2Const*
_output_shapes
: *
dtype0*
valueB
 *дp}?2	
Const_2Я
AssignMovingAvg/sub/xConst**
_class 
loc:@AssignMovingAvg/2736143*
_output_shapes
: *
dtype0*
valueB
 *  А?2
AssignMovingAvg/sub/x░
AssignMovingAvg/subSubAssignMovingAvg/sub/x:output:0Const_2:output:0*
T0**
_class 
loc:@AssignMovingAvg/2736143*
_output_shapes
: 2
AssignMovingAvg/subФ
AssignMovingAvg/ReadVariableOpReadVariableOpassignmovingavg_2736143*
_output_shapes
: *
dtype02 
AssignMovingAvg/ReadVariableOp═
AssignMovingAvg/sub_1Sub&AssignMovingAvg/ReadVariableOp:value:0FusedBatchNormV3:batch_mean:0*
T0**
_class 
loc:@AssignMovingAvg/2736143*
_output_shapes
: 2
AssignMovingAvg/sub_1╢
AssignMovingAvg/mulMulAssignMovingAvg/sub_1:z:0AssignMovingAvg/sub:z:0*
T0**
_class 
loc:@AssignMovingAvg/2736143*
_output_shapes
: 2
AssignMovingAvg/mulГ
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOpassignmovingavg_2736143AssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp**
_class 
loc:@AssignMovingAvg/2736143*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOpе
AssignMovingAvg_1/sub/xConst*,
_class"
 loc:@AssignMovingAvg_1/2736150*
_output_shapes
: *
dtype0*
valueB
 *  А?2
AssignMovingAvg_1/sub/x╕
AssignMovingAvg_1/subSub AssignMovingAvg_1/sub/x:output:0Const_2:output:0*
T0*,
_class"
 loc:@AssignMovingAvg_1/2736150*
_output_shapes
: 2
AssignMovingAvg_1/subЪ
 AssignMovingAvg_1/ReadVariableOpReadVariableOpassignmovingavg_1_2736150*
_output_shapes
: *
dtype02"
 AssignMovingAvg_1/ReadVariableOp┘
AssignMovingAvg_1/sub_1Sub(AssignMovingAvg_1/ReadVariableOp:value:0!FusedBatchNormV3:batch_variance:0*
T0*,
_class"
 loc:@AssignMovingAvg_1/2736150*
_output_shapes
: 2
AssignMovingAvg_1/sub_1└
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub_1:z:0AssignMovingAvg_1/sub:z:0*
T0*,
_class"
 loc:@AssignMovingAvg_1/2736150*
_output_shapes
: 2
AssignMovingAvg_1/mulП
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOpassignmovingavg_1_2736150AssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*,
_class"
 loc:@AssignMovingAvg_1/2736150*
_output_shapes
 *
dtype02'
%AssignMovingAvg_1/AssignSubVariableOpи
IdentityIdentityFusedBatchNormV3:y:0$^AssignMovingAvg/AssignSubVariableOp^AssignMovingAvg/ReadVariableOp&^AssignMovingAvg_1/AssignSubVariableOp!^AssignMovingAvg_1/ReadVariableOp^ReadVariableOp^ReadVariableOp_1*
T0*1
_output_shapes
:         АА 2

Identity"
identityIdentity:output:0*@
_input_shapes/
-:         АА ::::2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp2D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp2 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:& "
 
_user_specified_nameinputs
д
є
*__inference_b_norm_3_layer_call_fn_2738233

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4
identityИвStatefulPartitionedCall╒
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4*
Tin	
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*0
_output_shapes
:         А*-
config_proto

GPU

CPU2*0J 8*N
fIRG
E__inference_b_norm_3_layer_call_and_return_conditional_losses_27365062
StatefulPartitionedCallЧ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*0
_output_shapes
:         А2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:         А::::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs
Ї
H
,__inference_dropout_28_layer_call_fn_2737733

inputs
identity║
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*/
_output_shapes
:         @@ *-
config_proto

GPU

CPU2*0J 8*P
fKRI
G__inference_dropout_28_layer_call_and_return_conditional_losses_27362432
PartitionedCallt
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:         @@ 2

Identity"
identityIdentity:output:0*.
_input_shapes
:         @@ :& "
 
_user_specified_nameinputs
┴
й
(__inference_conv_1_layer_call_fn_2735482

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2
identityИвStatefulPartitionedCallв
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*A
_output_shapes/
-:+                            *-
config_proto

GPU

CPU2*0J 8*L
fGRE
C__inference_conv_1_layer_call_and_return_conditional_losses_27354742
StatefulPartitionedCallи
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+                            2

Identity"
identityIdentity:output:0*H
_input_shapes7
5:+                            ::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs
╓
∙
V__inference_input_batch_normalization_layer_call_and_return_conditional_losses_2736180

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityИвFusedBatchNormV3/ReadVariableOpв!FusedBatchNormV3/ReadVariableOp_1вReadVariableOpвReadVariableOp_1^
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
ReadVariableOp_1з
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02!
FusedBatchNormV3/ReadVariableOpн
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02#
!FusedBatchNormV3/ReadVariableOp_1╠
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*M
_output_shapes;
9:         АА : : : : :*
epsilon%oГ:*
is_training( 2
FusedBatchNormV3S
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *дp}?2
Const▄
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*1
_output_shapes
:         АА 2

Identity"
identityIdentity:output:0*@
_input_shapes/
-:         АА ::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:& "
 
_user_specified_nameinputs
┐
ш
E__inference_b_norm_1_layer_call_and_return_conditional_losses_2736314

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityИвFusedBatchNormV3/ReadVariableOpв!FusedBatchNormV3/ReadVariableOp_1вReadVariableOpвReadVariableOp_1^
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
ReadVariableOp_1з
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02!
FusedBatchNormV3/ReadVariableOpн
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02#
!FusedBatchNormV3/ReadVariableOp_1╩
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:         @@ : : : : :*
epsilon%oГ:*
is_training( 2
FusedBatchNormV3S
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *дp}?2
Const┌
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*/
_output_shapes
:         @@ 2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:         @@ ::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:& "
 
_user_specified_nameinputs
╥
N
2__inference_max_pooling2d_60_layer_call_fn_2735462

inputs
identity█
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*J
_output_shapes8
6:4                                    *-
config_proto

GPU

CPU2*0J 8*V
fQRO
M__inference_max_pooling2d_60_layer_call_and_return_conditional_losses_27354562
PartitionedCallП
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4                                    2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4                                    :& "
 
_user_specified_nameinputs
▀
H
,__inference_dropout_29_layer_call_fn_2738459

inputs
identity│
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*(
_output_shapes
:         А*-
config_proto

GPU

CPU2*0J 8*P
fKRI
G__inference_dropout_29_layer_call_and_return_conditional_losses_27366792
PartitionedCallm
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:         А2

Identity"
identityIdentity:output:0*'
_input_shapes
:         А:& "
 
_user_specified_nameinputs
б
є
*__inference_b_norm_1_layer_call_fn_2737810

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4
identityИвStatefulPartitionedCall╘
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4*
Tin	
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*/
_output_shapes
:         @@ *-
config_proto

GPU

CPU2*0J 8*N
fIRG
E__inference_b_norm_1_layer_call_and_return_conditional_losses_27362922
StatefulPartitionedCallЦ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:         @@ 2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:         @@ ::::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs
╔
Д
;__inference_input_batch_normalization_layer_call_fn_2737614

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4
identityИвStatefulPartitionedCallч
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4*
Tin	
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*1
_output_shapes
:         АА *-
config_proto

GPU

CPU2*0J 8*_
fZRX
V__inference_input_batch_normalization_layer_call_and_return_conditional_losses_27361802
StatefulPartitionedCallШ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*1
_output_shapes
:         АА 2

Identity"
identityIdentity:output:0*@
_input_shapes/
-:         АА ::::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs
ч
H
,__inference_flatten_14_layer_call_fn_2738424

inputs
identity│
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*(
_output_shapes
:         А*-
config_proto

GPU

CPU2*0J 8*P
fKRI
G__inference_flatten_14_layer_call_and_return_conditional_losses_27366462
PartitionedCallm
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:         А2

Identity"
identityIdentity:output:0*/
_input_shapes
:         А:& "
 
_user_specified_nameinputs
А
ш
E__inference_b_norm_4_layer_call_and_return_conditional_losses_2736099

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityИвFusedBatchNormV3/ReadVariableOpв!FusedBatchNormV3/ReadVariableOp_1вReadVariableOpвReadVariableOp_1^
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

LogicalAndu
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:А*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:А*
dtype02
ReadVariableOp_1и
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:А*
dtype02!
FusedBatchNormV3/ReadVariableOpо
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:А*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1с
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,                           А:А:А:А:А:*
epsilon%oГ:*
is_training( 2
FusedBatchNormV3S
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *дp}?2
Constэ
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*B
_output_shapes0
.:,                           А2

Identity"
identityIdentity:output:0*Q
_input_shapes@
>:,                           А::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:& "
 
_user_specified_nameinputs
И$
Т
E__inference_b_norm_1_layer_call_and_return_conditional_losses_2737779

inputs
readvariableop_resource
readvariableop_1_resource
assignmovingavg_2737764
assignmovingavg_1_2737771
identityИв#AssignMovingAvg/AssignSubVariableOpвAssignMovingAvg/ReadVariableOpв%AssignMovingAvg_1/AssignSubVariableOpв AssignMovingAvg_1/ReadVariableOpвReadVariableOpвReadVariableOp_1^
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
Const_1Е
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0Const:output:0Const_1:output:0*
T0*
U0*K
_output_shapes9
7:         @@ : : : : :*
epsilon%oГ:2
FusedBatchNormV3W
Const_2Const*
_output_shapes
: *
dtype0*
valueB
 *дp}?2	
Const_2Я
AssignMovingAvg/sub/xConst**
_class 
loc:@AssignMovingAvg/2737764*
_output_shapes
: *
dtype0*
valueB
 *  А?2
AssignMovingAvg/sub/x░
AssignMovingAvg/subSubAssignMovingAvg/sub/x:output:0Const_2:output:0*
T0**
_class 
loc:@AssignMovingAvg/2737764*
_output_shapes
: 2
AssignMovingAvg/subФ
AssignMovingAvg/ReadVariableOpReadVariableOpassignmovingavg_2737764*
_output_shapes
: *
dtype02 
AssignMovingAvg/ReadVariableOp═
AssignMovingAvg/sub_1Sub&AssignMovingAvg/ReadVariableOp:value:0FusedBatchNormV3:batch_mean:0*
T0**
_class 
loc:@AssignMovingAvg/2737764*
_output_shapes
: 2
AssignMovingAvg/sub_1╢
AssignMovingAvg/mulMulAssignMovingAvg/sub_1:z:0AssignMovingAvg/sub:z:0*
T0**
_class 
loc:@AssignMovingAvg/2737764*
_output_shapes
: 2
AssignMovingAvg/mulГ
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOpassignmovingavg_2737764AssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp**
_class 
loc:@AssignMovingAvg/2737764*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOpе
AssignMovingAvg_1/sub/xConst*,
_class"
 loc:@AssignMovingAvg_1/2737771*
_output_shapes
: *
dtype0*
valueB
 *  А?2
AssignMovingAvg_1/sub/x╕
AssignMovingAvg_1/subSub AssignMovingAvg_1/sub/x:output:0Const_2:output:0*
T0*,
_class"
 loc:@AssignMovingAvg_1/2737771*
_output_shapes
: 2
AssignMovingAvg_1/subЪ
 AssignMovingAvg_1/ReadVariableOpReadVariableOpassignmovingavg_1_2737771*
_output_shapes
: *
dtype02"
 AssignMovingAvg_1/ReadVariableOp┘
AssignMovingAvg_1/sub_1Sub(AssignMovingAvg_1/ReadVariableOp:value:0!FusedBatchNormV3:batch_variance:0*
T0*,
_class"
 loc:@AssignMovingAvg_1/2737771*
_output_shapes
: 2
AssignMovingAvg_1/sub_1└
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub_1:z:0AssignMovingAvg_1/sub:z:0*
T0*,
_class"
 loc:@AssignMovingAvg_1/2737771*
_output_shapes
: 2
AssignMovingAvg_1/mulП
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOpassignmovingavg_1_2737771AssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*,
_class"
 loc:@AssignMovingAvg_1/2737771*
_output_shapes
 *
dtype02'
%AssignMovingAvg_1/AssignSubVariableOpж
IdentityIdentityFusedBatchNormV3:y:0$^AssignMovingAvg/AssignSubVariableOp^AssignMovingAvg/ReadVariableOp&^AssignMovingAvg_1/AssignSubVariableOp!^AssignMovingAvg_1/ReadVariableOp^ReadVariableOp^ReadVariableOp_1*
T0*/
_output_shapes
:         @@ 2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:         @@ ::::2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp2D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp2 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:& "
 
_user_specified_nameinputs
ь
D
(__inference_RELU_2_layer_call_fn_2738073

inputs
identity╢
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*/
_output_shapes
:           @*-
config_proto

GPU

CPU2*0J 8*L
fGRE
C__inference_RELU_2_layer_call_and_return_conditional_losses_27364392
PartitionedCallt
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:           @2

Identity"
identityIdentity:output:0*.
_input_shapes
:           @:& "
 
_user_specified_nameinputs
г
_
C__inference_RELU_3_layer_call_and_return_conditional_losses_2738238

inputs
identityW
ReluReluinputs*
T0*0
_output_shapes
:         А2
Reluo
IdentityIdentityRelu:activations:0*
T0*0
_output_shapes
:         А2

Identity"
identityIdentity:output:0*/
_input_shapes
:         А:& "
 
_user_specified_nameinputs
∙
Д
;__inference_input_batch_normalization_layer_call_fn_2737679

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4
identityИвStatefulPartitionedCallў
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4*
Tin	
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*A
_output_shapes/
-:+                            *-
config_proto

GPU

CPU2*0J 8*_
fZRX
V__inference_input_batch_normalization_layer_call_and_return_conditional_losses_27354122
StatefulPartitionedCallи
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+                            2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+                            ::::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs
ї
ш
E__inference_b_norm_2_layer_call_and_return_conditional_losses_2735771

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityИвFusedBatchNormV3/ReadVariableOpв!FusedBatchNormV3/ReadVariableOp_1вReadVariableOpвReadVariableOp_1^
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
ReadVariableOp_1з
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOpн
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1▄
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+                           @:@:@:@:@:*
epsilon%oГ:*
is_training( 2
FusedBatchNormV3S
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *дp}?2
Constь
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*A
_output_shapes/
-:+                           @2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+                           @::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:& "
 
_user_specified_nameinputs
═$
Т
E__inference_b_norm_4_layer_call_and_return_conditional_losses_2736068

inputs
readvariableop_resource
readvariableop_1_resource
assignmovingavg_2736053
assignmovingavg_1_2736060
identityИв#AssignMovingAvg/AssignSubVariableOpвAssignMovingAvg/ReadVariableOpв%AssignMovingAvg_1/AssignSubVariableOpв AssignMovingAvg_1/ReadVariableOpвReadVariableOpвReadVariableOp_1^
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

LogicalAndu
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:А*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:А*
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
Const_1Ь
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0Const:output:0Const_1:output:0*
T0*
U0*b
_output_shapesP
N:,                           А:А:А:А:А:*
epsilon%oГ:2
FusedBatchNormV3W
Const_2Const*
_output_shapes
: *
dtype0*
valueB
 *дp}?2	
Const_2Я
AssignMovingAvg/sub/xConst**
_class 
loc:@AssignMovingAvg/2736053*
_output_shapes
: *
dtype0*
valueB
 *  А?2
AssignMovingAvg/sub/x░
AssignMovingAvg/subSubAssignMovingAvg/sub/x:output:0Const_2:output:0*
T0**
_class 
loc:@AssignMovingAvg/2736053*
_output_shapes
: 2
AssignMovingAvg/subХ
AssignMovingAvg/ReadVariableOpReadVariableOpassignmovingavg_2736053*
_output_shapes	
:А*
dtype02 
AssignMovingAvg/ReadVariableOp╬
AssignMovingAvg/sub_1Sub&AssignMovingAvg/ReadVariableOp:value:0FusedBatchNormV3:batch_mean:0*
T0**
_class 
loc:@AssignMovingAvg/2736053*
_output_shapes	
:А2
AssignMovingAvg/sub_1╖
AssignMovingAvg/mulMulAssignMovingAvg/sub_1:z:0AssignMovingAvg/sub:z:0*
T0**
_class 
loc:@AssignMovingAvg/2736053*
_output_shapes	
:А2
AssignMovingAvg/mulГ
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOpassignmovingavg_2736053AssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp**
_class 
loc:@AssignMovingAvg/2736053*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOpе
AssignMovingAvg_1/sub/xConst*,
_class"
 loc:@AssignMovingAvg_1/2736060*
_output_shapes
: *
dtype0*
valueB
 *  А?2
AssignMovingAvg_1/sub/x╕
AssignMovingAvg_1/subSub AssignMovingAvg_1/sub/x:output:0Const_2:output:0*
T0*,
_class"
 loc:@AssignMovingAvg_1/2736060*
_output_shapes
: 2
AssignMovingAvg_1/subЫ
 AssignMovingAvg_1/ReadVariableOpReadVariableOpassignmovingavg_1_2736060*
_output_shapes	
:А*
dtype02"
 AssignMovingAvg_1/ReadVariableOp┌
AssignMovingAvg_1/sub_1Sub(AssignMovingAvg_1/ReadVariableOp:value:0!FusedBatchNormV3:batch_variance:0*
T0*,
_class"
 loc:@AssignMovingAvg_1/2736060*
_output_shapes	
:А2
AssignMovingAvg_1/sub_1┴
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub_1:z:0AssignMovingAvg_1/sub:z:0*
T0*,
_class"
 loc:@AssignMovingAvg_1/2736060*
_output_shapes	
:А2
AssignMovingAvg_1/mulП
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOpassignmovingavg_1_2736060AssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*,
_class"
 loc:@AssignMovingAvg_1/2736060*
_output_shapes
 *
dtype02'
%AssignMovingAvg_1/AssignSubVariableOp╣
IdentityIdentityFusedBatchNormV3:y:0$^AssignMovingAvg/AssignSubVariableOp^AssignMovingAvg/ReadVariableOp&^AssignMovingAvg_1/AssignSubVariableOp!^AssignMovingAvg_1/ReadVariableOp^ReadVariableOp^ReadVariableOp_1*
T0*B
_output_shapes0
.:,                           А2

Identity"
identityIdentity:output:0*Q
_input_shapes@
>:,                           А::::2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp2D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp2 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:& "
 
_user_specified_nameinputs
╞
№	
,__inference_awsome_net_layer_call_fn_2737491

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
identityИвStatefulPartitionedCallВ
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4statefulpartitionedcall_args_5statefulpartitionedcall_args_6statefulpartitionedcall_args_7statefulpartitionedcall_args_8statefulpartitionedcall_args_9statefulpartitionedcall_args_10statefulpartitionedcall_args_11statefulpartitionedcall_args_12statefulpartitionedcall_args_13statefulpartitionedcall_args_14statefulpartitionedcall_args_15statefulpartitionedcall_args_16statefulpartitionedcall_args_17statefulpartitionedcall_args_18statefulpartitionedcall_args_19statefulpartitionedcall_args_20statefulpartitionedcall_args_21statefulpartitionedcall_args_22statefulpartitionedcall_args_23statefulpartitionedcall_args_24statefulpartitionedcall_args_25statefulpartitionedcall_args_26statefulpartitionedcall_args_27statefulpartitionedcall_args_28statefulpartitionedcall_args_29statefulpartitionedcall_args_30statefulpartitionedcall_args_31statefulpartitionedcall_args_32*,
Tin%
#2!*
Tout
2*,
_gradient_op_typePartitionedCallUnused*(
_output_shapes
:         ╚*-
config_proto

GPU

CPU2*0J 8*P
fKRI
G__inference_awsome_net_layer_call_and_return_conditional_losses_27368382
StatefulPartitionedCallП
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:         ╚2

Identity"
identityIdentity:output:0*▓
_input_shapesа
Э:         АА::::::::::::::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs
─
й
(__inference_conv_4_layer_call_fn_2735974

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2
identityИвStatefulPartitionedCallг
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*B
_output_shapes0
.:,                           А*-
config_proto

GPU

CPU2*0J 8*L
fGRE
C__inference_conv_4_layer_call_and_return_conditional_losses_27359662
StatefulPartitionedCallй
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*B
_output_shapes0
.:,                           А2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:,                           А::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs
ь
f
G__inference_dropout_28_layer_call_and_return_conditional_losses_2736238

inputs
identityИa
dropout/rateConst*
_output_shapes
: *
dtype0*
valueB
 *  А>2
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
 *  А?2
dropout/random_uniform/max╝
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*/
_output_shapes
:         @@ *
dtype02&
$dropout/random_uniform/RandomUniformк
dropout/random_uniform/subSub#dropout/random_uniform/max:output:0#dropout/random_uniform/min:output:0*
T0*
_output_shapes
: 2
dropout/random_uniform/sub╚
dropout/random_uniform/mulMul-dropout/random_uniform/RandomUniform:output:0dropout/random_uniform/sub:z:0*
T0*/
_output_shapes
:         @@ 2
dropout/random_uniform/mul╢
dropout/random_uniformAdddropout/random_uniform/mul:z:0#dropout/random_uniform/min:output:0*
T0*/
_output_shapes
:         @@ 2
dropout/random_uniformc
dropout/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  А?2
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
 *  А?2
dropout/truediv/x{
dropout/truedivRealDivdropout/truediv/x:output:0dropout/sub:z:0*
T0*
_output_shapes
: 2
dropout/truedivй
dropout/GreaterEqualGreaterEqualdropout/random_uniform:z:0dropout/rate:output:0*
T0*/
_output_shapes
:         @@ 2
dropout/GreaterEqualx
dropout/mulMulinputsdropout/truediv:z:0*
T0*/
_output_shapes
:         @@ 2
dropout/mulЗ
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:         @@ 2
dropout/CastВ
dropout/mul_1Muldropout/mul:z:0dropout/Cast:y:0*
T0*/
_output_shapes
:         @@ 2
dropout/mul_1m
IdentityIdentitydropout/mul_1:z:0*
T0*/
_output_shapes
:         @@ 2

Identity"
identityIdentity:output:0*.
_input_shapes
:         @@ :& "
 
_user_specified_nameinputs
╛$
Т
E__inference_b_norm_1_layer_call_and_return_conditional_losses_2735576

inputs
readvariableop_resource
readvariableop_1_resource
assignmovingavg_2735561
assignmovingavg_1_2735568
identityИв#AssignMovingAvg/AssignSubVariableOpвAssignMovingAvg/ReadVariableOpв%AssignMovingAvg_1/AssignSubVariableOpв AssignMovingAvg_1/ReadVariableOpвReadVariableOpвReadVariableOp_1^
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
Const_1Ч
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0Const:output:0Const_1:output:0*
T0*
U0*]
_output_shapesK
I:+                            : : : : :*
epsilon%oГ:2
FusedBatchNormV3W
Const_2Const*
_output_shapes
: *
dtype0*
valueB
 *дp}?2	
Const_2Я
AssignMovingAvg/sub/xConst**
_class 
loc:@AssignMovingAvg/2735561*
_output_shapes
: *
dtype0*
valueB
 *  А?2
AssignMovingAvg/sub/x░
AssignMovingAvg/subSubAssignMovingAvg/sub/x:output:0Const_2:output:0*
T0**
_class 
loc:@AssignMovingAvg/2735561*
_output_shapes
: 2
AssignMovingAvg/subФ
AssignMovingAvg/ReadVariableOpReadVariableOpassignmovingavg_2735561*
_output_shapes
: *
dtype02 
AssignMovingAvg/ReadVariableOp═
AssignMovingAvg/sub_1Sub&AssignMovingAvg/ReadVariableOp:value:0FusedBatchNormV3:batch_mean:0*
T0**
_class 
loc:@AssignMovingAvg/2735561*
_output_shapes
: 2
AssignMovingAvg/sub_1╢
AssignMovingAvg/mulMulAssignMovingAvg/sub_1:z:0AssignMovingAvg/sub:z:0*
T0**
_class 
loc:@AssignMovingAvg/2735561*
_output_shapes
: 2
AssignMovingAvg/mulГ
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOpassignmovingavg_2735561AssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp**
_class 
loc:@AssignMovingAvg/2735561*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOpе
AssignMovingAvg_1/sub/xConst*,
_class"
 loc:@AssignMovingAvg_1/2735568*
_output_shapes
: *
dtype0*
valueB
 *  А?2
AssignMovingAvg_1/sub/x╕
AssignMovingAvg_1/subSub AssignMovingAvg_1/sub/x:output:0Const_2:output:0*
T0*,
_class"
 loc:@AssignMovingAvg_1/2735568*
_output_shapes
: 2
AssignMovingAvg_1/subЪ
 AssignMovingAvg_1/ReadVariableOpReadVariableOpassignmovingavg_1_2735568*
_output_shapes
: *
dtype02"
 AssignMovingAvg_1/ReadVariableOp┘
AssignMovingAvg_1/sub_1Sub(AssignMovingAvg_1/ReadVariableOp:value:0!FusedBatchNormV3:batch_variance:0*
T0*,
_class"
 loc:@AssignMovingAvg_1/2735568*
_output_shapes
: 2
AssignMovingAvg_1/sub_1└
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub_1:z:0AssignMovingAvg_1/sub:z:0*
T0*,
_class"
 loc:@AssignMovingAvg_1/2735568*
_output_shapes
: 2
AssignMovingAvg_1/mulП
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOpassignmovingavg_1_2735568AssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*,
_class"
 loc:@AssignMovingAvg_1/2735568*
_output_shapes
 *
dtype02'
%AssignMovingAvg_1/AssignSubVariableOp╕
IdentityIdentityFusedBatchNormV3:y:0$^AssignMovingAvg/AssignSubVariableOp^AssignMovingAvg/ReadVariableOp&^AssignMovingAvg_1/AssignSubVariableOp!^AssignMovingAvg_1/ReadVariableOp^ReadVariableOp^ReadVariableOp_1*
T0*A
_output_shapes/
-:+                            2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+                            ::::2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp2D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp2 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:& "
 
_user_specified_nameinputs
д
є
*__inference_b_norm_4_layer_call_fn_2738329

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4
identityИвStatefulPartitionedCall╒
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4*
Tin	
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*0
_output_shapes
:         А*-
config_proto

GPU

CPU2*0J 8*N
fIRG
E__inference_b_norm_4_layer_call_and_return_conditional_losses_27366022
StatefulPartitionedCallЧ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*0
_output_shapes
:         А2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:         А::::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs
┐
ш
E__inference_b_norm_2_layer_call_and_return_conditional_losses_2736410

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityИвFusedBatchNormV3/ReadVariableOpв!FusedBatchNormV3/ReadVariableOp_1вReadVariableOpвReadVariableOp_1^
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
ReadVariableOp_1з
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOpн
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1╩
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:           @:@:@:@:@:*
epsilon%oГ:*
is_training( 2
FusedBatchNormV3S
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *дp}?2
Const┌
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*/
_output_shapes
:           @2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:           @::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:& "
 
_user_specified_nameinputs
к
c
G__inference_input_RELU_layer_call_and_return_conditional_losses_2737693

inputs
identityX
ReluReluinputs*
T0*1
_output_shapes
:         АА 2
Relup
IdentityIdentityRelu:activations:0*
T0*1
_output_shapes
:         АА 2

Identity"
identityIdentity:output:0*0
_input_shapes
:         АА :& "
 
_user_specified_nameinputs
ї
ш
E__inference_b_norm_1_layer_call_and_return_conditional_losses_2737875

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityИвFusedBatchNormV3/ReadVariableOpв!FusedBatchNormV3/ReadVariableOp_1вReadVariableOpвReadVariableOp_1^
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
ReadVariableOp_1з
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02!
FusedBatchNormV3/ReadVariableOpн
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02#
!FusedBatchNormV3/ReadVariableOp_1▄
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+                            : : : : :*
epsilon%oГ:*
is_training( 2
FusedBatchNormV3S
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *дp}?2
Constь
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*A
_output_shapes/
-:+                            2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+                            ::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:& "
 
_user_specified_nameinputs
м
f
G__inference_dropout_29_layer_call_and_return_conditional_losses_2738444

inputs
identityИa
dropout/rateConst*
_output_shapes
: *
dtype0*
valueB
 *  А>2
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
 *  А?2
dropout/random_uniform/max╡
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*(
_output_shapes
:         А*
dtype02&
$dropout/random_uniform/RandomUniformк
dropout/random_uniform/subSub#dropout/random_uniform/max:output:0#dropout/random_uniform/min:output:0*
T0*
_output_shapes
: 2
dropout/random_uniform/sub┴
dropout/random_uniform/mulMul-dropout/random_uniform/RandomUniform:output:0dropout/random_uniform/sub:z:0*
T0*(
_output_shapes
:         А2
dropout/random_uniform/mulп
dropout/random_uniformAdddropout/random_uniform/mul:z:0#dropout/random_uniform/min:output:0*
T0*(
_output_shapes
:         А2
dropout/random_uniformc
dropout/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  А?2
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
 *  А?2
dropout/truediv/x{
dropout/truedivRealDivdropout/truediv/x:output:0dropout/sub:z:0*
T0*
_output_shapes
: 2
dropout/truedivв
dropout/GreaterEqualGreaterEqualdropout/random_uniform:z:0dropout/rate:output:0*
T0*(
_output_shapes
:         А2
dropout/GreaterEqualq
dropout/mulMulinputsdropout/truediv:z:0*
T0*(
_output_shapes
:         А2
dropout/mulА
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:         А2
dropout/Cast{
dropout/mul_1Muldropout/mul:z:0dropout/Cast:y:0*
T0*(
_output_shapes
:         А2
dropout/mul_1f
IdentityIdentitydropout/mul_1:z:0*
T0*(
_output_shapes
:         А2

Identity"
identityIdentity:output:0*'
_input_shapes
:         А:& "
 
_user_specified_nameinputs
╧$
г
V__inference_input_batch_normalization_layer_call_and_return_conditional_losses_2737648

inputs
readvariableop_resource
readvariableop_1_resource
assignmovingavg_2737633
assignmovingavg_1_2737640
identityИв#AssignMovingAvg/AssignSubVariableOpвAssignMovingAvg/ReadVariableOpв%AssignMovingAvg_1/AssignSubVariableOpв AssignMovingAvg_1/ReadVariableOpвReadVariableOpвReadVariableOp_1^
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
Const_1Ч
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0Const:output:0Const_1:output:0*
T0*
U0*]
_output_shapesK
I:+                            : : : : :*
epsilon%oГ:2
FusedBatchNormV3W
Const_2Const*
_output_shapes
: *
dtype0*
valueB
 *дp}?2	
Const_2Я
AssignMovingAvg/sub/xConst**
_class 
loc:@AssignMovingAvg/2737633*
_output_shapes
: *
dtype0*
valueB
 *  А?2
AssignMovingAvg/sub/x░
AssignMovingAvg/subSubAssignMovingAvg/sub/x:output:0Const_2:output:0*
T0**
_class 
loc:@AssignMovingAvg/2737633*
_output_shapes
: 2
AssignMovingAvg/subФ
AssignMovingAvg/ReadVariableOpReadVariableOpassignmovingavg_2737633*
_output_shapes
: *
dtype02 
AssignMovingAvg/ReadVariableOp═
AssignMovingAvg/sub_1Sub&AssignMovingAvg/ReadVariableOp:value:0FusedBatchNormV3:batch_mean:0*
T0**
_class 
loc:@AssignMovingAvg/2737633*
_output_shapes
: 2
AssignMovingAvg/sub_1╢
AssignMovingAvg/mulMulAssignMovingAvg/sub_1:z:0AssignMovingAvg/sub:z:0*
T0**
_class 
loc:@AssignMovingAvg/2737633*
_output_shapes
: 2
AssignMovingAvg/mulГ
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOpassignmovingavg_2737633AssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp**
_class 
loc:@AssignMovingAvg/2737633*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOpе
AssignMovingAvg_1/sub/xConst*,
_class"
 loc:@AssignMovingAvg_1/2737640*
_output_shapes
: *
dtype0*
valueB
 *  А?2
AssignMovingAvg_1/sub/x╕
AssignMovingAvg_1/subSub AssignMovingAvg_1/sub/x:output:0Const_2:output:0*
T0*,
_class"
 loc:@AssignMovingAvg_1/2737640*
_output_shapes
: 2
AssignMovingAvg_1/subЪ
 AssignMovingAvg_1/ReadVariableOpReadVariableOpassignmovingavg_1_2737640*
_output_shapes
: *
dtype02"
 AssignMovingAvg_1/ReadVariableOp┘
AssignMovingAvg_1/sub_1Sub(AssignMovingAvg_1/ReadVariableOp:value:0!FusedBatchNormV3:batch_variance:0*
T0*,
_class"
 loc:@AssignMovingAvg_1/2737640*
_output_shapes
: 2
AssignMovingAvg_1/sub_1└
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub_1:z:0AssignMovingAvg_1/sub:z:0*
T0*,
_class"
 loc:@AssignMovingAvg_1/2737640*
_output_shapes
: 2
AssignMovingAvg_1/mulП
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOpassignmovingavg_1_2737640AssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*,
_class"
 loc:@AssignMovingAvg_1/2737640*
_output_shapes
 *
dtype02'
%AssignMovingAvg_1/AssignSubVariableOp╕
IdentityIdentityFusedBatchNormV3:y:0$^AssignMovingAvg/AssignSubVariableOp^AssignMovingAvg/ReadVariableOp&^AssignMovingAvg_1/AssignSubVariableOp!^AssignMovingAvg_1/ReadVariableOp^ReadVariableOp^ReadVariableOp_1*
T0*A
_output_shapes/
-:+                            2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+                            ::::2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp2D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp2 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:& "
 
_user_specified_nameinputs
Ж
∙
V__inference_input_batch_normalization_layer_call_and_return_conditional_losses_2735443

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityИвFusedBatchNormV3/ReadVariableOpв!FusedBatchNormV3/ReadVariableOp_1вReadVariableOpвReadVariableOp_1^
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
ReadVariableOp_1з
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02!
FusedBatchNormV3/ReadVariableOpн
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02#
!FusedBatchNormV3/ReadVariableOp_1▄
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+                            : : : : :*
epsilon%oГ:*
is_training( 2
FusedBatchNormV3S
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *дp}?2
Constь
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*A
_output_shapes/
-:+                            2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+                            ::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:& "
 
_user_specified_nameinputs
═$
Т
E__inference_b_norm_3_layer_call_and_return_conditional_losses_2735904

inputs
readvariableop_resource
readvariableop_1_resource
assignmovingavg_2735889
assignmovingavg_1_2735896
identityИв#AssignMovingAvg/AssignSubVariableOpвAssignMovingAvg/ReadVariableOpв%AssignMovingAvg_1/AssignSubVariableOpв AssignMovingAvg_1/ReadVariableOpвReadVariableOpвReadVariableOp_1^
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

LogicalAndu
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:А*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:А*
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
Const_1Ь
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0Const:output:0Const_1:output:0*
T0*
U0*b
_output_shapesP
N:,                           А:А:А:А:А:*
epsilon%oГ:2
FusedBatchNormV3W
Const_2Const*
_output_shapes
: *
dtype0*
valueB
 *дp}?2	
Const_2Я
AssignMovingAvg/sub/xConst**
_class 
loc:@AssignMovingAvg/2735889*
_output_shapes
: *
dtype0*
valueB
 *  А?2
AssignMovingAvg/sub/x░
AssignMovingAvg/subSubAssignMovingAvg/sub/x:output:0Const_2:output:0*
T0**
_class 
loc:@AssignMovingAvg/2735889*
_output_shapes
: 2
AssignMovingAvg/subХ
AssignMovingAvg/ReadVariableOpReadVariableOpassignmovingavg_2735889*
_output_shapes	
:А*
dtype02 
AssignMovingAvg/ReadVariableOp╬
AssignMovingAvg/sub_1Sub&AssignMovingAvg/ReadVariableOp:value:0FusedBatchNormV3:batch_mean:0*
T0**
_class 
loc:@AssignMovingAvg/2735889*
_output_shapes	
:А2
AssignMovingAvg/sub_1╖
AssignMovingAvg/mulMulAssignMovingAvg/sub_1:z:0AssignMovingAvg/sub:z:0*
T0**
_class 
loc:@AssignMovingAvg/2735889*
_output_shapes	
:А2
AssignMovingAvg/mulГ
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOpassignmovingavg_2735889AssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp**
_class 
loc:@AssignMovingAvg/2735889*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOpе
AssignMovingAvg_1/sub/xConst*,
_class"
 loc:@AssignMovingAvg_1/2735896*
_output_shapes
: *
dtype0*
valueB
 *  А?2
AssignMovingAvg_1/sub/x╕
AssignMovingAvg_1/subSub AssignMovingAvg_1/sub/x:output:0Const_2:output:0*
T0*,
_class"
 loc:@AssignMovingAvg_1/2735896*
_output_shapes
: 2
AssignMovingAvg_1/subЫ
 AssignMovingAvg_1/ReadVariableOpReadVariableOpassignmovingavg_1_2735896*
_output_shapes	
:А*
dtype02"
 AssignMovingAvg_1/ReadVariableOp┌
AssignMovingAvg_1/sub_1Sub(AssignMovingAvg_1/ReadVariableOp:value:0!FusedBatchNormV3:batch_variance:0*
T0*,
_class"
 loc:@AssignMovingAvg_1/2735896*
_output_shapes	
:А2
AssignMovingAvg_1/sub_1┴
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub_1:z:0AssignMovingAvg_1/sub:z:0*
T0*,
_class"
 loc:@AssignMovingAvg_1/2735896*
_output_shapes	
:А2
AssignMovingAvg_1/mulП
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOpassignmovingavg_1_2735896AssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*,
_class"
 loc:@AssignMovingAvg_1/2735896*
_output_shapes
 *
dtype02'
%AssignMovingAvg_1/AssignSubVariableOp╣
IdentityIdentityFusedBatchNormV3:y:0$^AssignMovingAvg/AssignSubVariableOp^AssignMovingAvg/ReadVariableOp&^AssignMovingAvg_1/AssignSubVariableOp!^AssignMovingAvg_1/ReadVariableOp^ReadVariableOp^ReadVariableOp_1*
T0*B
_output_shapes0
.:,                           А2

Identity"
identityIdentity:output:0*Q
_input_shapes@
>:,                           А::::2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp2D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp2 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:& "
 
_user_specified_nameinputs
╫
є
*__inference_b_norm_1_layer_call_fn_2737893

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4
identityИвStatefulPartitionedCallц
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4*
Tin	
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*A
_output_shapes/
-:+                            *-
config_proto

GPU

CPU2*0J 8*N
fIRG
E__inference_b_norm_1_layer_call_and_return_conditional_losses_27356072
StatefulPartitionedCallи
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+                            2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+                            ::::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs
д
є
*__inference_b_norm_4_layer_call_fn_2738320

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4
identityИвStatefulPartitionedCall╒
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4*
Tin	
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*0
_output_shapes
:         А*-
config_proto

GPU

CPU2*0J 8*N
fIRG
E__inference_b_norm_4_layer_call_and_return_conditional_losses_27365802
StatefulPartitionedCallЧ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*0
_output_shapes
:         А2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:         А::::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs
╖
i
M__inference_max_pooling2d_57_layer_call_and_return_conditional_losses_2735784

inputs
identityн
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4                                    *
ksize
*
paddingVALID*
strides
2	
MaxPoolЗ
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4                                    2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4                                    :& "
 
_user_specified_nameinputs
г
_
C__inference_RELU_3_layer_call_and_return_conditional_losses_2736535

inputs
identityW
ReluReluinputs*
T0*0
_output_shapes
:         А2
Reluo
IdentityIdentityRelu:activations:0*
T0*0
_output_shapes
:         А2

Identity"
identityIdentity:output:0*/
_input_shapes
:         А:& "
 
_user_specified_nameinputs
И$
Т
E__inference_b_norm_2_layer_call_and_return_conditional_losses_2736388

inputs
readvariableop_resource
readvariableop_1_resource
assignmovingavg_2736373
assignmovingavg_1_2736380
identityИв#AssignMovingAvg/AssignSubVariableOpвAssignMovingAvg/ReadVariableOpв%AssignMovingAvg_1/AssignSubVariableOpв AssignMovingAvg_1/ReadVariableOpвReadVariableOpвReadVariableOp_1^
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
Const_1Е
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0Const:output:0Const_1:output:0*
T0*
U0*K
_output_shapes9
7:           @:@:@:@:@:*
epsilon%oГ:2
FusedBatchNormV3W
Const_2Const*
_output_shapes
: *
dtype0*
valueB
 *дp}?2	
Const_2Я
AssignMovingAvg/sub/xConst**
_class 
loc:@AssignMovingAvg/2736373*
_output_shapes
: *
dtype0*
valueB
 *  А?2
AssignMovingAvg/sub/x░
AssignMovingAvg/subSubAssignMovingAvg/sub/x:output:0Const_2:output:0*
T0**
_class 
loc:@AssignMovingAvg/2736373*
_output_shapes
: 2
AssignMovingAvg/subФ
AssignMovingAvg/ReadVariableOpReadVariableOpassignmovingavg_2736373*
_output_shapes
:@*
dtype02 
AssignMovingAvg/ReadVariableOp═
AssignMovingAvg/sub_1Sub&AssignMovingAvg/ReadVariableOp:value:0FusedBatchNormV3:batch_mean:0*
T0**
_class 
loc:@AssignMovingAvg/2736373*
_output_shapes
:@2
AssignMovingAvg/sub_1╢
AssignMovingAvg/mulMulAssignMovingAvg/sub_1:z:0AssignMovingAvg/sub:z:0*
T0**
_class 
loc:@AssignMovingAvg/2736373*
_output_shapes
:@2
AssignMovingAvg/mulГ
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOpassignmovingavg_2736373AssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp**
_class 
loc:@AssignMovingAvg/2736373*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOpе
AssignMovingAvg_1/sub/xConst*,
_class"
 loc:@AssignMovingAvg_1/2736380*
_output_shapes
: *
dtype0*
valueB
 *  А?2
AssignMovingAvg_1/sub/x╕
AssignMovingAvg_1/subSub AssignMovingAvg_1/sub/x:output:0Const_2:output:0*
T0*,
_class"
 loc:@AssignMovingAvg_1/2736380*
_output_shapes
: 2
AssignMovingAvg_1/subЪ
 AssignMovingAvg_1/ReadVariableOpReadVariableOpassignmovingavg_1_2736380*
_output_shapes
:@*
dtype02"
 AssignMovingAvg_1/ReadVariableOp┘
AssignMovingAvg_1/sub_1Sub(AssignMovingAvg_1/ReadVariableOp:value:0!FusedBatchNormV3:batch_variance:0*
T0*,
_class"
 loc:@AssignMovingAvg_1/2736380*
_output_shapes
:@2
AssignMovingAvg_1/sub_1└
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub_1:z:0AssignMovingAvg_1/sub:z:0*
T0*,
_class"
 loc:@AssignMovingAvg_1/2736380*
_output_shapes
:@2
AssignMovingAvg_1/mulП
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOpassignmovingavg_1_2736380AssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*,
_class"
 loc:@AssignMovingAvg_1/2736380*
_output_shapes
 *
dtype02'
%AssignMovingAvg_1/AssignSubVariableOpж
IdentityIdentityFusedBatchNormV3:y:0$^AssignMovingAvg/AssignSubVariableOp^AssignMovingAvg/ReadVariableOp&^AssignMovingAvg_1/AssignSubVariableOp!^AssignMovingAvg_1/ReadVariableOp^ReadVariableOp^ReadVariableOp_1*
T0*/
_output_shapes
:           @2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:           @::::2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp2D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp2 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:& "
 
_user_specified_nameinputs
┐
ш
E__inference_b_norm_2_layer_call_and_return_conditional_losses_2738045

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityИвFusedBatchNormV3/ReadVariableOpв!FusedBatchNormV3/ReadVariableOp_1вReadVariableOpвReadVariableOp_1^
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
ReadVariableOp_1з
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOpн
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1╩
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:           @:@:@:@:@:*
epsilon%oГ:*
is_training( 2
FusedBatchNormV3S
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *дp}?2
Const┌
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*/
_output_shapes
:           @2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:           @::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:& "
 
_user_specified_nameinputs
Тp
З
G__inference_awsome_net_layer_call_and_return_conditional_losses_2736775
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
identityИв b_norm_1/StatefulPartitionedCallв b_norm_2/StatefulPartitionedCallв b_norm_3/StatefulPartitionedCallв b_norm_4/StatefulPartitionedCallвconv_1/StatefulPartitionedCallвconv_2/StatefulPartitionedCallвconv_3/StatefulPartitionedCallвconv_4/StatefulPartitionedCallв1input_batch_normalization/StatefulPartitionedCallв"input_conv/StatefulPartitionedCallв1output_class_distribution/StatefulPartitionedCall╠
"input_conv/StatefulPartitionedCallStatefulPartitionedCallinput_conv_input)input_conv_statefulpartitionedcall_args_1)input_conv_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*1
_output_shapes
:         АА *-
config_proto

GPU

CPU2*0J 8*P
fKRI
G__inference_input_conv_layer_call_and_return_conditional_losses_27353102$
"input_conv/StatefulPartitionedCallи
1input_batch_normalization/StatefulPartitionedCallStatefulPartitionedCall+input_conv/StatefulPartitionedCall:output:08input_batch_normalization_statefulpartitionedcall_args_18input_batch_normalization_statefulpartitionedcall_args_28input_batch_normalization_statefulpartitionedcall_args_38input_batch_normalization_statefulpartitionedcall_args_4*
Tin	
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*1
_output_shapes
:         АА *-
config_proto

GPU

CPU2*0J 8*_
fZRX
V__inference_input_batch_normalization_layer_call_and_return_conditional_losses_273618023
1input_batch_normalization/StatefulPartitionedCallЖ
input_RELU/PartitionedCallPartitionedCall:input_batch_normalization/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*1
_output_shapes
:         АА *-
config_proto

GPU

CPU2*0J 8*P
fKRI
G__inference_input_RELU_layer_call_and_return_conditional_losses_27362092
input_RELU/PartitionedCall 
 max_pooling2d_60/PartitionedCallPartitionedCall#input_RELU/PartitionedCall:output:0*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*/
_output_shapes
:         @@ *-
config_proto

GPU

CPU2*0J 8*V
fQRO
M__inference_max_pooling2d_60_layer_call_and_return_conditional_losses_27354562"
 max_pooling2d_60/PartitionedCallє
dropout_28/PartitionedCallPartitionedCall)max_pooling2d_60/PartitionedCall:output:0*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*/
_output_shapes
:         @@ *-
config_proto

GPU

CPU2*0J 8*P
fKRI
G__inference_dropout_28_layer_call_and_return_conditional_losses_27362432
dropout_28/PartitionedCall╔
conv_1/StatefulPartitionedCallStatefulPartitionedCall#dropout_28/PartitionedCall:output:0%conv_1_statefulpartitionedcall_args_1%conv_1_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*/
_output_shapes
:         @@ *-
config_proto

GPU

CPU2*0J 8*L
fGRE
C__inference_conv_1_layer_call_and_return_conditional_losses_27354742 
conv_1/StatefulPartitionedCallл
 b_norm_1/StatefulPartitionedCallStatefulPartitionedCall'conv_1/StatefulPartitionedCall:output:0'b_norm_1_statefulpartitionedcall_args_1'b_norm_1_statefulpartitionedcall_args_2'b_norm_1_statefulpartitionedcall_args_3'b_norm_1_statefulpartitionedcall_args_4*
Tin	
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*/
_output_shapes
:         @@ *-
config_proto

GPU

CPU2*0J 8*N
fIRG
E__inference_b_norm_1_layer_call_and_return_conditional_losses_27363142"
 b_norm_1/StatefulPartitionedCallч
RELU_1/PartitionedCallPartitionedCall)b_norm_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*/
_output_shapes
:         @@ *-
config_proto

GPU

CPU2*0J 8*L
fGRE
C__inference_RELU_1_layer_call_and_return_conditional_losses_27363432
RELU_1/PartitionedCall√
 max_pooling2d_56/PartitionedCallPartitionedCallRELU_1/PartitionedCall:output:0*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*/
_output_shapes
:            *-
config_proto

GPU

CPU2*0J 8*V
fQRO
M__inference_max_pooling2d_56_layer_call_and_return_conditional_losses_27356202"
 max_pooling2d_56/PartitionedCall╧
conv_2/StatefulPartitionedCallStatefulPartitionedCall)max_pooling2d_56/PartitionedCall:output:0%conv_2_statefulpartitionedcall_args_1%conv_2_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*/
_output_shapes
:           @*-
config_proto

GPU

CPU2*0J 8*L
fGRE
C__inference_conv_2_layer_call_and_return_conditional_losses_27356382 
conv_2/StatefulPartitionedCallл
 b_norm_2/StatefulPartitionedCallStatefulPartitionedCall'conv_2/StatefulPartitionedCall:output:0'b_norm_2_statefulpartitionedcall_args_1'b_norm_2_statefulpartitionedcall_args_2'b_norm_2_statefulpartitionedcall_args_3'b_norm_2_statefulpartitionedcall_args_4*
Tin	
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*/
_output_shapes
:           @*-
config_proto

GPU

CPU2*0J 8*N
fIRG
E__inference_b_norm_2_layer_call_and_return_conditional_losses_27364102"
 b_norm_2/StatefulPartitionedCallч
RELU_2/PartitionedCallPartitionedCall)b_norm_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*/
_output_shapes
:           @*-
config_proto

GPU

CPU2*0J 8*L
fGRE
C__inference_RELU_2_layer_call_and_return_conditional_losses_27364392
RELU_2/PartitionedCall√
 max_pooling2d_57/PartitionedCallPartitionedCallRELU_2/PartitionedCall:output:0*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*/
_output_shapes
:         @*-
config_proto

GPU

CPU2*0J 8*V
fQRO
M__inference_max_pooling2d_57_layer_call_and_return_conditional_losses_27357842"
 max_pooling2d_57/PartitionedCall╨
conv_3/StatefulPartitionedCallStatefulPartitionedCall)max_pooling2d_57/PartitionedCall:output:0%conv_3_statefulpartitionedcall_args_1%conv_3_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*0
_output_shapes
:         А*-
config_proto

GPU

CPU2*0J 8*L
fGRE
C__inference_conv_3_layer_call_and_return_conditional_losses_27358022 
conv_3/StatefulPartitionedCallм
 b_norm_3/StatefulPartitionedCallStatefulPartitionedCall'conv_3/StatefulPartitionedCall:output:0'b_norm_3_statefulpartitionedcall_args_1'b_norm_3_statefulpartitionedcall_args_2'b_norm_3_statefulpartitionedcall_args_3'b_norm_3_statefulpartitionedcall_args_4*
Tin	
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*0
_output_shapes
:         А*-
config_proto

GPU

CPU2*0J 8*N
fIRG
E__inference_b_norm_3_layer_call_and_return_conditional_losses_27365062"
 b_norm_3/StatefulPartitionedCallш
RELU_3/PartitionedCallPartitionedCall)b_norm_3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*0
_output_shapes
:         А*-
config_proto

GPU

CPU2*0J 8*L
fGRE
C__inference_RELU_3_layer_call_and_return_conditional_losses_27365352
RELU_3/PartitionedCall№
 max_pooling2d_58/PartitionedCallPartitionedCallRELU_3/PartitionedCall:output:0*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*0
_output_shapes
:         А*-
config_proto

GPU

CPU2*0J 8*V
fQRO
M__inference_max_pooling2d_58_layer_call_and_return_conditional_losses_27359482"
 max_pooling2d_58/PartitionedCall╨
conv_4/StatefulPartitionedCallStatefulPartitionedCall)max_pooling2d_58/PartitionedCall:output:0%conv_4_statefulpartitionedcall_args_1%conv_4_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*0
_output_shapes
:         А*-
config_proto

GPU

CPU2*0J 8*L
fGRE
C__inference_conv_4_layer_call_and_return_conditional_losses_27359662 
conv_4/StatefulPartitionedCallм
 b_norm_4/StatefulPartitionedCallStatefulPartitionedCall'conv_4/StatefulPartitionedCall:output:0'b_norm_4_statefulpartitionedcall_args_1'b_norm_4_statefulpartitionedcall_args_2'b_norm_4_statefulpartitionedcall_args_3'b_norm_4_statefulpartitionedcall_args_4*
Tin	
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*0
_output_shapes
:         А*-
config_proto

GPU

CPU2*0J 8*N
fIRG
E__inference_b_norm_4_layer_call_and_return_conditional_losses_27366022"
 b_norm_4/StatefulPartitionedCallш
RELU_4/PartitionedCallPartitionedCall)b_norm_4/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*0
_output_shapes
:         А*-
config_proto

GPU

CPU2*0J 8*L
fGRE
C__inference_RELU_4_layer_call_and_return_conditional_losses_27366312
RELU_4/PartitionedCall№
 max_pooling2d_59/PartitionedCallPartitionedCallRELU_4/PartitionedCall:output:0*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*0
_output_shapes
:         А*-
config_proto

GPU

CPU2*0J 8*V
fQRO
M__inference_max_pooling2d_59_layer_call_and_return_conditional_losses_27361122"
 max_pooling2d_59/PartitionedCallь
flatten_14/PartitionedCallPartitionedCall)max_pooling2d_59/PartitionedCall:output:0*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*(
_output_shapes
:         А*-
config_proto

GPU

CPU2*0J 8*P
fKRI
G__inference_flatten_14_layer_call_and_return_conditional_losses_27366462
flatten_14/PartitionedCallц
dropout_29/PartitionedCallPartitionedCall#flatten_14/PartitionedCall:output:0*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*(
_output_shapes
:         А*-
config_proto

GPU

CPU2*0J 8*P
fKRI
G__inference_dropout_29_layer_call_and_return_conditional_losses_27366792
dropout_29/PartitionedCallб
1output_class_distribution/StatefulPartitionedCallStatefulPartitionedCall#dropout_29/PartitionedCall:output:08output_class_distribution_statefulpartitionedcall_args_18output_class_distribution_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*(
_output_shapes
:         ╚*-
config_proto

GPU

CPU2*0J 8*_
fZRX
V__inference_output_class_distribution_layer_call_and_return_conditional_losses_273670223
1output_class_distribution/StatefulPartitionedCallм
IdentityIdentity:output_class_distribution/StatefulPartitionedCall:output:0!^b_norm_1/StatefulPartitionedCall!^b_norm_2/StatefulPartitionedCall!^b_norm_3/StatefulPartitionedCall!^b_norm_4/StatefulPartitionedCall^conv_1/StatefulPartitionedCall^conv_2/StatefulPartitionedCall^conv_3/StatefulPartitionedCall^conv_4/StatefulPartitionedCall2^input_batch_normalization/StatefulPartitionedCall#^input_conv/StatefulPartitionedCall2^output_class_distribution/StatefulPartitionedCall*
T0*(
_output_shapes
:         ╚2

Identity"
identityIdentity:output:0*▓
_input_shapesа
Э:         АА::::::::::::::::::::::::::::::::2D
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
╩
ш
E__inference_b_norm_3_layer_call_and_return_conditional_losses_2736506

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityИвFusedBatchNormV3/ReadVariableOpв!FusedBatchNormV3/ReadVariableOp_1вReadVariableOpвReadVariableOp_1^
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

LogicalAndu
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:А*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:А*
dtype02
ReadVariableOp_1и
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:А*
dtype02!
FusedBatchNormV3/ReadVariableOpо
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:А*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1╧
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:         А:А:А:А:А:*
epsilon%oГ:*
is_training( 2
FusedBatchNormV3S
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *дp}?2
Const█
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*0
_output_shapes
:         А2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:         А::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:& "
 
_user_specified_nameinputs
ь
f
G__inference_dropout_28_layer_call_and_return_conditional_losses_2737718

inputs
identityИa
dropout/rateConst*
_output_shapes
: *
dtype0*
valueB
 *  А>2
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
 *  А?2
dropout/random_uniform/max╝
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*/
_output_shapes
:         @@ *
dtype02&
$dropout/random_uniform/RandomUniformк
dropout/random_uniform/subSub#dropout/random_uniform/max:output:0#dropout/random_uniform/min:output:0*
T0*
_output_shapes
: 2
dropout/random_uniform/sub╚
dropout/random_uniform/mulMul-dropout/random_uniform/RandomUniform:output:0dropout/random_uniform/sub:z:0*
T0*/
_output_shapes
:         @@ 2
dropout/random_uniform/mul╢
dropout/random_uniformAdddropout/random_uniform/mul:z:0#dropout/random_uniform/min:output:0*
T0*/
_output_shapes
:         @@ 2
dropout/random_uniformc
dropout/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  А?2
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
 *  А?2
dropout/truediv/x{
dropout/truedivRealDivdropout/truediv/x:output:0dropout/sub:z:0*
T0*
_output_shapes
: 2
dropout/truedivй
dropout/GreaterEqualGreaterEqualdropout/random_uniform:z:0dropout/rate:output:0*
T0*/
_output_shapes
:         @@ 2
dropout/GreaterEqualx
dropout/mulMulinputsdropout/truediv:z:0*
T0*/
_output_shapes
:         @@ 2
dropout/mulЗ
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:         @@ 2
dropout/CastВ
dropout/mul_1Muldropout/mul:z:0dropout/Cast:y:0*
T0*/
_output_shapes
:         @@ 2
dropout/mul_1m
IdentityIdentitydropout/mul_1:z:0*
T0*/
_output_shapes
:         @@ 2

Identity"
identityIdentity:output:0*.
_input_shapes
:         @@ :& "
 
_user_specified_nameinputs
╣
e
G__inference_dropout_28_layer_call_and_return_conditional_losses_2737723

inputs

identity_1b
IdentityIdentityinputs*
T0*/
_output_shapes
:         @@ 2

Identityq

Identity_1IdentityIdentity:output:0*
T0*/
_output_shapes
:         @@ 2

Identity_1"!

identity_1Identity_1:output:0*.
_input_shapes
:         @@ :& "
 
_user_specified_nameinputs
б
є
*__inference_b_norm_1_layer_call_fn_2737819

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4
identityИвStatefulPartitionedCall╘
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4*
Tin	
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*/
_output_shapes
:         @@ *-
config_proto

GPU

CPU2*0J 8*N
fIRG
E__inference_b_norm_1_layer_call_and_return_conditional_losses_27363142
StatefulPartitionedCallЦ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:         @@ 2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:         @@ ::::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs
П
c
G__inference_flatten_14_layer_call_and_return_conditional_losses_2736646

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"       2
Consth
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:         А2	
Reshapee
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:         А2

Identity"
identityIdentity:output:0*/
_input_shapes
:         А:& "
 
_user_specified_nameinputs
Яs
╤
G__inference_awsome_net_layer_call_and_return_conditional_losses_2736715
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
identityИв b_norm_1/StatefulPartitionedCallв b_norm_2/StatefulPartitionedCallв b_norm_3/StatefulPartitionedCallв b_norm_4/StatefulPartitionedCallвconv_1/StatefulPartitionedCallвconv_2/StatefulPartitionedCallвconv_3/StatefulPartitionedCallвconv_4/StatefulPartitionedCallв"dropout_28/StatefulPartitionedCallв"dropout_29/StatefulPartitionedCallв1input_batch_normalization/StatefulPartitionedCallв"input_conv/StatefulPartitionedCallв1output_class_distribution/StatefulPartitionedCall╠
"input_conv/StatefulPartitionedCallStatefulPartitionedCallinput_conv_input)input_conv_statefulpartitionedcall_args_1)input_conv_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*1
_output_shapes
:         АА *-
config_proto

GPU

CPU2*0J 8*P
fKRI
G__inference_input_conv_layer_call_and_return_conditional_losses_27353102$
"input_conv/StatefulPartitionedCallи
1input_batch_normalization/StatefulPartitionedCallStatefulPartitionedCall+input_conv/StatefulPartitionedCall:output:08input_batch_normalization_statefulpartitionedcall_args_18input_batch_normalization_statefulpartitionedcall_args_28input_batch_normalization_statefulpartitionedcall_args_38input_batch_normalization_statefulpartitionedcall_args_4*
Tin	
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*1
_output_shapes
:         АА *-
config_proto

GPU

CPU2*0J 8*_
fZRX
V__inference_input_batch_normalization_layer_call_and_return_conditional_losses_273615823
1input_batch_normalization/StatefulPartitionedCallЖ
input_RELU/PartitionedCallPartitionedCall:input_batch_normalization/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*1
_output_shapes
:         АА *-
config_proto

GPU

CPU2*0J 8*P
fKRI
G__inference_input_RELU_layer_call_and_return_conditional_losses_27362092
input_RELU/PartitionedCall 
 max_pooling2d_60/PartitionedCallPartitionedCall#input_RELU/PartitionedCall:output:0*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*/
_output_shapes
:         @@ *-
config_proto

GPU

CPU2*0J 8*V
fQRO
M__inference_max_pooling2d_60_layer_call_and_return_conditional_losses_27354562"
 max_pooling2d_60/PartitionedCallЛ
"dropout_28/StatefulPartitionedCallStatefulPartitionedCall)max_pooling2d_60/PartitionedCall:output:0*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*/
_output_shapes
:         @@ *-
config_proto

GPU

CPU2*0J 8*P
fKRI
G__inference_dropout_28_layer_call_and_return_conditional_losses_27362382$
"dropout_28/StatefulPartitionedCall╤
conv_1/StatefulPartitionedCallStatefulPartitionedCall+dropout_28/StatefulPartitionedCall:output:0%conv_1_statefulpartitionedcall_args_1%conv_1_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*/
_output_shapes
:         @@ *-
config_proto

GPU

CPU2*0J 8*L
fGRE
C__inference_conv_1_layer_call_and_return_conditional_losses_27354742 
conv_1/StatefulPartitionedCallл
 b_norm_1/StatefulPartitionedCallStatefulPartitionedCall'conv_1/StatefulPartitionedCall:output:0'b_norm_1_statefulpartitionedcall_args_1'b_norm_1_statefulpartitionedcall_args_2'b_norm_1_statefulpartitionedcall_args_3'b_norm_1_statefulpartitionedcall_args_4*
Tin	
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*/
_output_shapes
:         @@ *-
config_proto

GPU

CPU2*0J 8*N
fIRG
E__inference_b_norm_1_layer_call_and_return_conditional_losses_27362922"
 b_norm_1/StatefulPartitionedCallч
RELU_1/PartitionedCallPartitionedCall)b_norm_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*/
_output_shapes
:         @@ *-
config_proto

GPU

CPU2*0J 8*L
fGRE
C__inference_RELU_1_layer_call_and_return_conditional_losses_27363432
RELU_1/PartitionedCall√
 max_pooling2d_56/PartitionedCallPartitionedCallRELU_1/PartitionedCall:output:0*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*/
_output_shapes
:            *-
config_proto

GPU

CPU2*0J 8*V
fQRO
M__inference_max_pooling2d_56_layer_call_and_return_conditional_losses_27356202"
 max_pooling2d_56/PartitionedCall╧
conv_2/StatefulPartitionedCallStatefulPartitionedCall)max_pooling2d_56/PartitionedCall:output:0%conv_2_statefulpartitionedcall_args_1%conv_2_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*/
_output_shapes
:           @*-
config_proto

GPU

CPU2*0J 8*L
fGRE
C__inference_conv_2_layer_call_and_return_conditional_losses_27356382 
conv_2/StatefulPartitionedCallл
 b_norm_2/StatefulPartitionedCallStatefulPartitionedCall'conv_2/StatefulPartitionedCall:output:0'b_norm_2_statefulpartitionedcall_args_1'b_norm_2_statefulpartitionedcall_args_2'b_norm_2_statefulpartitionedcall_args_3'b_norm_2_statefulpartitionedcall_args_4*
Tin	
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*/
_output_shapes
:           @*-
config_proto

GPU

CPU2*0J 8*N
fIRG
E__inference_b_norm_2_layer_call_and_return_conditional_losses_27363882"
 b_norm_2/StatefulPartitionedCallч
RELU_2/PartitionedCallPartitionedCall)b_norm_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*/
_output_shapes
:           @*-
config_proto

GPU

CPU2*0J 8*L
fGRE
C__inference_RELU_2_layer_call_and_return_conditional_losses_27364392
RELU_2/PartitionedCall√
 max_pooling2d_57/PartitionedCallPartitionedCallRELU_2/PartitionedCall:output:0*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*/
_output_shapes
:         @*-
config_proto

GPU

CPU2*0J 8*V
fQRO
M__inference_max_pooling2d_57_layer_call_and_return_conditional_losses_27357842"
 max_pooling2d_57/PartitionedCall╨
conv_3/StatefulPartitionedCallStatefulPartitionedCall)max_pooling2d_57/PartitionedCall:output:0%conv_3_statefulpartitionedcall_args_1%conv_3_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*0
_output_shapes
:         А*-
config_proto

GPU

CPU2*0J 8*L
fGRE
C__inference_conv_3_layer_call_and_return_conditional_losses_27358022 
conv_3/StatefulPartitionedCallм
 b_norm_3/StatefulPartitionedCallStatefulPartitionedCall'conv_3/StatefulPartitionedCall:output:0'b_norm_3_statefulpartitionedcall_args_1'b_norm_3_statefulpartitionedcall_args_2'b_norm_3_statefulpartitionedcall_args_3'b_norm_3_statefulpartitionedcall_args_4*
Tin	
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*0
_output_shapes
:         А*-
config_proto

GPU

CPU2*0J 8*N
fIRG
E__inference_b_norm_3_layer_call_and_return_conditional_losses_27364842"
 b_norm_3/StatefulPartitionedCallш
RELU_3/PartitionedCallPartitionedCall)b_norm_3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*0
_output_shapes
:         А*-
config_proto

GPU

CPU2*0J 8*L
fGRE
C__inference_RELU_3_layer_call_and_return_conditional_losses_27365352
RELU_3/PartitionedCall№
 max_pooling2d_58/PartitionedCallPartitionedCallRELU_3/PartitionedCall:output:0*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*0
_output_shapes
:         А*-
config_proto

GPU

CPU2*0J 8*V
fQRO
M__inference_max_pooling2d_58_layer_call_and_return_conditional_losses_27359482"
 max_pooling2d_58/PartitionedCall╨
conv_4/StatefulPartitionedCallStatefulPartitionedCall)max_pooling2d_58/PartitionedCall:output:0%conv_4_statefulpartitionedcall_args_1%conv_4_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*0
_output_shapes
:         А*-
config_proto

GPU

CPU2*0J 8*L
fGRE
C__inference_conv_4_layer_call_and_return_conditional_losses_27359662 
conv_4/StatefulPartitionedCallм
 b_norm_4/StatefulPartitionedCallStatefulPartitionedCall'conv_4/StatefulPartitionedCall:output:0'b_norm_4_statefulpartitionedcall_args_1'b_norm_4_statefulpartitionedcall_args_2'b_norm_4_statefulpartitionedcall_args_3'b_norm_4_statefulpartitionedcall_args_4*
Tin	
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*0
_output_shapes
:         А*-
config_proto

GPU

CPU2*0J 8*N
fIRG
E__inference_b_norm_4_layer_call_and_return_conditional_losses_27365802"
 b_norm_4/StatefulPartitionedCallш
RELU_4/PartitionedCallPartitionedCall)b_norm_4/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*0
_output_shapes
:         А*-
config_proto

GPU

CPU2*0J 8*L
fGRE
C__inference_RELU_4_layer_call_and_return_conditional_losses_27366312
RELU_4/PartitionedCall№
 max_pooling2d_59/PartitionedCallPartitionedCallRELU_4/PartitionedCall:output:0*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*0
_output_shapes
:         А*-
config_proto

GPU

CPU2*0J 8*V
fQRO
M__inference_max_pooling2d_59_layer_call_and_return_conditional_losses_27361122"
 max_pooling2d_59/PartitionedCallь
flatten_14/PartitionedCallPartitionedCall)max_pooling2d_59/PartitionedCall:output:0*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*(
_output_shapes
:         А*-
config_proto

GPU

CPU2*0J 8*P
fKRI
G__inference_flatten_14_layer_call_and_return_conditional_losses_27366462
flatten_14/PartitionedCallг
"dropout_29/StatefulPartitionedCallStatefulPartitionedCall#flatten_14/PartitionedCall:output:0#^dropout_28/StatefulPartitionedCall*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*(
_output_shapes
:         А*-
config_proto

GPU

CPU2*0J 8*P
fKRI
G__inference_dropout_29_layer_call_and_return_conditional_losses_27366742$
"dropout_29/StatefulPartitionedCallй
1output_class_distribution/StatefulPartitionedCallStatefulPartitionedCall+dropout_29/StatefulPartitionedCall:output:08output_class_distribution_statefulpartitionedcall_args_18output_class_distribution_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*(
_output_shapes
:         ╚*-
config_proto

GPU

CPU2*0J 8*_
fZRX
V__inference_output_class_distribution_layer_call_and_return_conditional_losses_273670223
1output_class_distribution/StatefulPartitionedCallЎ
IdentityIdentity:output_class_distribution/StatefulPartitionedCall:output:0!^b_norm_1/StatefulPartitionedCall!^b_norm_2/StatefulPartitionedCall!^b_norm_3/StatefulPartitionedCall!^b_norm_4/StatefulPartitionedCall^conv_1/StatefulPartitionedCall^conv_2/StatefulPartitionedCall^conv_3/StatefulPartitionedCall^conv_4/StatefulPartitionedCall#^dropout_28/StatefulPartitionedCall#^dropout_29/StatefulPartitionedCall2^input_batch_normalization/StatefulPartitionedCall#^input_conv/StatefulPartitionedCall2^output_class_distribution/StatefulPartitionedCall*
T0*(
_output_shapes
:         ╚2

Identity"
identityIdentity:output:0*▓
_input_shapesа
Э:         АА::::::::::::::::::::::::::::::::2D
 b_norm_1/StatefulPartitionedCall b_norm_1/StatefulPartitionedCall2D
 b_norm_2/StatefulPartitionedCall b_norm_2/StatefulPartitionedCall2D
 b_norm_3/StatefulPartitionedCall b_norm_3/StatefulPartitionedCall2D
 b_norm_4/StatefulPartitionedCall b_norm_4/StatefulPartitionedCall2@
conv_1/StatefulPartitionedCallconv_1/StatefulPartitionedCall2@
conv_2/StatefulPartitionedCallconv_2/StatefulPartitionedCall2@
conv_3/StatefulPartitionedCallconv_3/StatefulPartitionedCall2@
conv_4/StatefulPartitionedCallconv_4/StatefulPartitionedCall2H
"dropout_28/StatefulPartitionedCall"dropout_28/StatefulPartitionedCall2H
"dropout_29/StatefulPartitionedCall"dropout_29/StatefulPartitionedCall2f
1input_batch_normalization/StatefulPartitionedCall1input_batch_normalization/StatefulPartitionedCall2H
"input_conv/StatefulPartitionedCall"input_conv/StatefulPartitionedCall2f
1output_class_distribution/StatefulPartitionedCall1output_class_distribution/StatefulPartitionedCall:0 ,
*
_user_specified_nameinput_conv_input
ўС
к%
 __inference__traced_save_2738749
file_prefix3
/savev2_input_conv_14_kernel_read_readvariableop1
-savev2_input_conv_14_bias_read_readvariableopA
=savev2_input_batch_normalization_14_gamma_read_readvariableop@
<savev2_input_batch_normalization_14_beta_read_readvariableopG
Csavev2_input_batch_normalization_14_moving_mean_read_readvariableopK
Gsavev2_input_batch_normalization_14_moving_variance_read_readvariableop/
+savev2_conv_1_14_kernel_read_readvariableop-
)savev2_conv_1_14_bias_read_readvariableop0
,savev2_b_norm_1_14_gamma_read_readvariableop/
+savev2_b_norm_1_14_beta_read_readvariableop6
2savev2_b_norm_1_14_moving_mean_read_readvariableop:
6savev2_b_norm_1_14_moving_variance_read_readvariableop/
+savev2_conv_2_14_kernel_read_readvariableop-
)savev2_conv_2_14_bias_read_readvariableop0
,savev2_b_norm_2_14_gamma_read_readvariableop/
+savev2_b_norm_2_14_beta_read_readvariableop6
2savev2_b_norm_2_14_moving_mean_read_readvariableop:
6savev2_b_norm_2_14_moving_variance_read_readvariableop.
*savev2_conv_3_9_kernel_read_readvariableop,
(savev2_conv_3_9_bias_read_readvariableop/
+savev2_b_norm_3_9_gamma_read_readvariableop.
*savev2_b_norm_3_9_beta_read_readvariableop5
1savev2_b_norm_3_9_moving_mean_read_readvariableop9
5savev2_b_norm_3_9_moving_variance_read_readvariableop.
*savev2_conv_4_4_kernel_read_readvariableop,
(savev2_conv_4_4_bias_read_readvariableop/
+savev2_b_norm_4_4_gamma_read_readvariableop.
*savev2_b_norm_4_4_beta_read_readvariableop5
1savev2_b_norm_4_4_moving_mean_read_readvariableop9
5savev2_b_norm_4_4_moving_variance_read_readvariableopB
>savev2_output_class_distribution_14_kernel_read_readvariableop@
<savev2_output_class_distribution_14_bias_read_readvariableop(
$savev2_adam_iter_read_readvariableop	*
&savev2_adam_beta_1_read_readvariableop*
&savev2_adam_beta_2_read_readvariableop)
%savev2_adam_decay_read_readvariableop1
-savev2_adam_learning_rate_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop:
6savev2_adam_input_conv_14_kernel_m_read_readvariableop8
4savev2_adam_input_conv_14_bias_m_read_readvariableopH
Dsavev2_adam_input_batch_normalization_14_gamma_m_read_readvariableopG
Csavev2_adam_input_batch_normalization_14_beta_m_read_readvariableop6
2savev2_adam_conv_1_14_kernel_m_read_readvariableop4
0savev2_adam_conv_1_14_bias_m_read_readvariableop7
3savev2_adam_b_norm_1_14_gamma_m_read_readvariableop6
2savev2_adam_b_norm_1_14_beta_m_read_readvariableop6
2savev2_adam_conv_2_14_kernel_m_read_readvariableop4
0savev2_adam_conv_2_14_bias_m_read_readvariableop7
3savev2_adam_b_norm_2_14_gamma_m_read_readvariableop6
2savev2_adam_b_norm_2_14_beta_m_read_readvariableop5
1savev2_adam_conv_3_9_kernel_m_read_readvariableop3
/savev2_adam_conv_3_9_bias_m_read_readvariableop6
2savev2_adam_b_norm_3_9_gamma_m_read_readvariableop5
1savev2_adam_b_norm_3_9_beta_m_read_readvariableop5
1savev2_adam_conv_4_4_kernel_m_read_readvariableop3
/savev2_adam_conv_4_4_bias_m_read_readvariableop6
2savev2_adam_b_norm_4_4_gamma_m_read_readvariableop5
1savev2_adam_b_norm_4_4_beta_m_read_readvariableopI
Esavev2_adam_output_class_distribution_14_kernel_m_read_readvariableopG
Csavev2_adam_output_class_distribution_14_bias_m_read_readvariableop:
6savev2_adam_input_conv_14_kernel_v_read_readvariableop8
4savev2_adam_input_conv_14_bias_v_read_readvariableopH
Dsavev2_adam_input_batch_normalization_14_gamma_v_read_readvariableopG
Csavev2_adam_input_batch_normalization_14_beta_v_read_readvariableop6
2savev2_adam_conv_1_14_kernel_v_read_readvariableop4
0savev2_adam_conv_1_14_bias_v_read_readvariableop7
3savev2_adam_b_norm_1_14_gamma_v_read_readvariableop6
2savev2_adam_b_norm_1_14_beta_v_read_readvariableop6
2savev2_adam_conv_2_14_kernel_v_read_readvariableop4
0savev2_adam_conv_2_14_bias_v_read_readvariableop7
3savev2_adam_b_norm_2_14_gamma_v_read_readvariableop6
2savev2_adam_b_norm_2_14_beta_v_read_readvariableop5
1savev2_adam_conv_3_9_kernel_v_read_readvariableop3
/savev2_adam_conv_3_9_bias_v_read_readvariableop6
2savev2_adam_b_norm_3_9_gamma_v_read_readvariableop5
1savev2_adam_b_norm_3_9_beta_v_read_readvariableop5
1savev2_adam_conv_4_4_kernel_v_read_readvariableop3
/savev2_adam_conv_4_4_bias_v_read_readvariableop6
2savev2_adam_b_norm_4_4_gamma_v_read_readvariableop5
1savev2_adam_b_norm_4_4_beta_v_read_readvariableopI
Esavev2_adam_output_class_distribution_14_kernel_v_read_readvariableopG
Csavev2_adam_output_class_distribution_14_bias_v_read_readvariableop
savev2_1_const

identity_1ИвMergeV2CheckpointsвSaveV2вSaveV2_1е
StringJoin/inputs_1Const"/device:CPU:0*
_output_shapes
: *
dtype0*<
value3B1 B+_temp_0a8636bbc4a14f108e4b93104fde503f/part2
StringJoin/inputs_1Б

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
ShardedFilename/shardж
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilename┐.
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:S*
dtype0*╤-
value╟-B─-SB6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-1/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-1/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-1/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-3/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-3/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-3/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-5/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-5/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-5/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-7/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-7/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-7/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-7/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-8/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-8/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-9/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-9/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-9/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-9/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-10/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-10/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-1/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-3/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-5/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-7/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-7/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-8/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-8/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-9/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-9/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-10/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-10/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-1/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-3/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-5/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-7/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-7/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-8/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-8/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-9/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-9/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-10/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-10/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE2
SaveV2/tensor_names▒
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:S*
dtype0*╗
value▒BоSB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
SaveV2/shape_and_slices┘#
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0/savev2_input_conv_14_kernel_read_readvariableop-savev2_input_conv_14_bias_read_readvariableop=savev2_input_batch_normalization_14_gamma_read_readvariableop<savev2_input_batch_normalization_14_beta_read_readvariableopCsavev2_input_batch_normalization_14_moving_mean_read_readvariableopGsavev2_input_batch_normalization_14_moving_variance_read_readvariableop+savev2_conv_1_14_kernel_read_readvariableop)savev2_conv_1_14_bias_read_readvariableop,savev2_b_norm_1_14_gamma_read_readvariableop+savev2_b_norm_1_14_beta_read_readvariableop2savev2_b_norm_1_14_moving_mean_read_readvariableop6savev2_b_norm_1_14_moving_variance_read_readvariableop+savev2_conv_2_14_kernel_read_readvariableop)savev2_conv_2_14_bias_read_readvariableop,savev2_b_norm_2_14_gamma_read_readvariableop+savev2_b_norm_2_14_beta_read_readvariableop2savev2_b_norm_2_14_moving_mean_read_readvariableop6savev2_b_norm_2_14_moving_variance_read_readvariableop*savev2_conv_3_9_kernel_read_readvariableop(savev2_conv_3_9_bias_read_readvariableop+savev2_b_norm_3_9_gamma_read_readvariableop*savev2_b_norm_3_9_beta_read_readvariableop1savev2_b_norm_3_9_moving_mean_read_readvariableop5savev2_b_norm_3_9_moving_variance_read_readvariableop*savev2_conv_4_4_kernel_read_readvariableop(savev2_conv_4_4_bias_read_readvariableop+savev2_b_norm_4_4_gamma_read_readvariableop*savev2_b_norm_4_4_beta_read_readvariableop1savev2_b_norm_4_4_moving_mean_read_readvariableop5savev2_b_norm_4_4_moving_variance_read_readvariableop>savev2_output_class_distribution_14_kernel_read_readvariableop<savev2_output_class_distribution_14_bias_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop6savev2_adam_input_conv_14_kernel_m_read_readvariableop4savev2_adam_input_conv_14_bias_m_read_readvariableopDsavev2_adam_input_batch_normalization_14_gamma_m_read_readvariableopCsavev2_adam_input_batch_normalization_14_beta_m_read_readvariableop2savev2_adam_conv_1_14_kernel_m_read_readvariableop0savev2_adam_conv_1_14_bias_m_read_readvariableop3savev2_adam_b_norm_1_14_gamma_m_read_readvariableop2savev2_adam_b_norm_1_14_beta_m_read_readvariableop2savev2_adam_conv_2_14_kernel_m_read_readvariableop0savev2_adam_conv_2_14_bias_m_read_readvariableop3savev2_adam_b_norm_2_14_gamma_m_read_readvariableop2savev2_adam_b_norm_2_14_beta_m_read_readvariableop1savev2_adam_conv_3_9_kernel_m_read_readvariableop/savev2_adam_conv_3_9_bias_m_read_readvariableop2savev2_adam_b_norm_3_9_gamma_m_read_readvariableop1savev2_adam_b_norm_3_9_beta_m_read_readvariableop1savev2_adam_conv_4_4_kernel_m_read_readvariableop/savev2_adam_conv_4_4_bias_m_read_readvariableop2savev2_adam_b_norm_4_4_gamma_m_read_readvariableop1savev2_adam_b_norm_4_4_beta_m_read_readvariableopEsavev2_adam_output_class_distribution_14_kernel_m_read_readvariableopCsavev2_adam_output_class_distribution_14_bias_m_read_readvariableop6savev2_adam_input_conv_14_kernel_v_read_readvariableop4savev2_adam_input_conv_14_bias_v_read_readvariableopDsavev2_adam_input_batch_normalization_14_gamma_v_read_readvariableopCsavev2_adam_input_batch_normalization_14_beta_v_read_readvariableop2savev2_adam_conv_1_14_kernel_v_read_readvariableop0savev2_adam_conv_1_14_bias_v_read_readvariableop3savev2_adam_b_norm_1_14_gamma_v_read_readvariableop2savev2_adam_b_norm_1_14_beta_v_read_readvariableop2savev2_adam_conv_2_14_kernel_v_read_readvariableop0savev2_adam_conv_2_14_bias_v_read_readvariableop3savev2_adam_b_norm_2_14_gamma_v_read_readvariableop2savev2_adam_b_norm_2_14_beta_v_read_readvariableop1savev2_adam_conv_3_9_kernel_v_read_readvariableop/savev2_adam_conv_3_9_bias_v_read_readvariableop2savev2_adam_b_norm_3_9_gamma_v_read_readvariableop1savev2_adam_b_norm_3_9_beta_v_read_readvariableop1savev2_adam_conv_4_4_kernel_v_read_readvariableop/savev2_adam_conv_4_4_bias_v_read_readvariableop2savev2_adam_b_norm_4_4_gamma_v_read_readvariableop1savev2_adam_b_norm_4_4_beta_v_read_readvariableopEsavev2_adam_output_class_distribution_14_kernel_v_read_readvariableopCsavev2_adam_output_class_distribution_14_bias_v_read_readvariableop"/device:CPU:0*
_output_shapes
 *a
dtypesW
U2S	2
SaveV2Г
ShardedFilename_1/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B :2
ShardedFilename_1/shardм
ShardedFilename_1ShardedFilenameStringJoin:output:0 ShardedFilename_1/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilename_1в
SaveV2_1/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*1
value(B&B_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2_1/tensor_namesО
SaveV2_1/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*
valueB
B 2
SaveV2_1/shape_and_slices╧
SaveV2_1SaveV2ShardedFilename_1:filename:0SaveV2_1/tensor_names:output:0"SaveV2_1/shape_and_slices:output:0savev2_1_const^SaveV2"/device:CPU:0*
_output_shapes
 *
dtypes
22

SaveV2_1у
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0ShardedFilename_1:filename:0^SaveV2	^SaveV2_1"/device:CPU:0*
N*
T0*
_output_shapes
:2(
&MergeV2Checkpoints/checkpoint_prefixesм
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix	^SaveV2_1"/device:CPU:0*
_output_shapes
 2
MergeV2Checkpointsr
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: 2

IdentityБ

Identity_1IdentityIdentity:output:0^MergeV2Checkpoints^SaveV2	^SaveV2_1*
T0*
_output_shapes
: 2

Identity_1"!

identity_1Identity_1:output:0*╫
_input_shapes┼
┬: : : : : : : :  : : : : : : @:@:@:@:@:@:@А:А:А:А:А:А:АА:А:А:А:А:А:
А╚:╚: : : : : : : : : : : :  : : : : @:@:@:@:@А:А:А:А:АА:А:А:А:
А╚:╚: : : : :  : : : : @:@:@:@:@А:А:А:А:АА:А:А:А:
А╚:╚: 2(
MergeV2CheckpointsMergeV2Checkpoints2
SaveV2SaveV22
SaveV2_1SaveV2_1:+ '
%
_user_specified_namefile_prefix
П
c
G__inference_flatten_14_layer_call_and_return_conditional_losses_2738419

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"       2
Consth
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:         А2	
Reshapee
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:         А2

Identity"
identityIdentity:output:0*/
_input_shapes
:         А:& "
 
_user_specified_nameinputs
╣
e
G__inference_dropout_28_layer_call_and_return_conditional_losses_2736243

inputs

identity_1b
IdentityIdentityinputs*
T0*/
_output_shapes
:         @@ 2

Identityq

Identity_1IdentityIdentity:output:0*
T0*/
_output_shapes
:         @@ 2

Identity_1"!

identity_1Identity_1:output:0*.
_input_shapes
:         @@ :& "
 
_user_specified_nameinputs
╛$
Т
E__inference_b_norm_1_layer_call_and_return_conditional_losses_2737853

inputs
readvariableop_resource
readvariableop_1_resource
assignmovingavg_2737838
assignmovingavg_1_2737845
identityИв#AssignMovingAvg/AssignSubVariableOpвAssignMovingAvg/ReadVariableOpв%AssignMovingAvg_1/AssignSubVariableOpв AssignMovingAvg_1/ReadVariableOpвReadVariableOpвReadVariableOp_1^
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
Const_1Ч
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0Const:output:0Const_1:output:0*
T0*
U0*]
_output_shapesK
I:+                            : : : : :*
epsilon%oГ:2
FusedBatchNormV3W
Const_2Const*
_output_shapes
: *
dtype0*
valueB
 *дp}?2	
Const_2Я
AssignMovingAvg/sub/xConst**
_class 
loc:@AssignMovingAvg/2737838*
_output_shapes
: *
dtype0*
valueB
 *  А?2
AssignMovingAvg/sub/x░
AssignMovingAvg/subSubAssignMovingAvg/sub/x:output:0Const_2:output:0*
T0**
_class 
loc:@AssignMovingAvg/2737838*
_output_shapes
: 2
AssignMovingAvg/subФ
AssignMovingAvg/ReadVariableOpReadVariableOpassignmovingavg_2737838*
_output_shapes
: *
dtype02 
AssignMovingAvg/ReadVariableOp═
AssignMovingAvg/sub_1Sub&AssignMovingAvg/ReadVariableOp:value:0FusedBatchNormV3:batch_mean:0*
T0**
_class 
loc:@AssignMovingAvg/2737838*
_output_shapes
: 2
AssignMovingAvg/sub_1╢
AssignMovingAvg/mulMulAssignMovingAvg/sub_1:z:0AssignMovingAvg/sub:z:0*
T0**
_class 
loc:@AssignMovingAvg/2737838*
_output_shapes
: 2
AssignMovingAvg/mulГ
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOpassignmovingavg_2737838AssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp**
_class 
loc:@AssignMovingAvg/2737838*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOpе
AssignMovingAvg_1/sub/xConst*,
_class"
 loc:@AssignMovingAvg_1/2737845*
_output_shapes
: *
dtype0*
valueB
 *  А?2
AssignMovingAvg_1/sub/x╕
AssignMovingAvg_1/subSub AssignMovingAvg_1/sub/x:output:0Const_2:output:0*
T0*,
_class"
 loc:@AssignMovingAvg_1/2737845*
_output_shapes
: 2
AssignMovingAvg_1/subЪ
 AssignMovingAvg_1/ReadVariableOpReadVariableOpassignmovingavg_1_2737845*
_output_shapes
: *
dtype02"
 AssignMovingAvg_1/ReadVariableOp┘
AssignMovingAvg_1/sub_1Sub(AssignMovingAvg_1/ReadVariableOp:value:0!FusedBatchNormV3:batch_variance:0*
T0*,
_class"
 loc:@AssignMovingAvg_1/2737845*
_output_shapes
: 2
AssignMovingAvg_1/sub_1└
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub_1:z:0AssignMovingAvg_1/sub:z:0*
T0*,
_class"
 loc:@AssignMovingAvg_1/2737845*
_output_shapes
: 2
AssignMovingAvg_1/mulП
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOpassignmovingavg_1_2737845AssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*,
_class"
 loc:@AssignMovingAvg_1/2737845*
_output_shapes
 *
dtype02'
%AssignMovingAvg_1/AssignSubVariableOp╕
IdentityIdentityFusedBatchNormV3:y:0$^AssignMovingAvg/AssignSubVariableOp^AssignMovingAvg/ReadVariableOp&^AssignMovingAvg_1/AssignSubVariableOp!^AssignMovingAvg_1/ReadVariableOp^ReadVariableOp^ReadVariableOp_1*
T0*A
_output_shapes/
-:+                            2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+                            ::::2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp2D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp2 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:& "
 
_user_specified_nameinputs
╥
N
2__inference_max_pooling2d_56_layer_call_fn_2735626

inputs
identity█
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*J
_output_shapes8
6:4                                    *-
config_proto

GPU

CPU2*0J 8*V
fQRO
M__inference_max_pooling2d_56_layer_call_and_return_conditional_losses_27356202
PartitionedCallП
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4                                    2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4                                    :& "
 
_user_specified_nameinputs
И$
Т
E__inference_b_norm_1_layer_call_and_return_conditional_losses_2736292

inputs
readvariableop_resource
readvariableop_1_resource
assignmovingavg_2736277
assignmovingavg_1_2736284
identityИв#AssignMovingAvg/AssignSubVariableOpвAssignMovingAvg/ReadVariableOpв%AssignMovingAvg_1/AssignSubVariableOpв AssignMovingAvg_1/ReadVariableOpвReadVariableOpвReadVariableOp_1^
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
Const_1Е
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0Const:output:0Const_1:output:0*
T0*
U0*K
_output_shapes9
7:         @@ : : : : :*
epsilon%oГ:2
FusedBatchNormV3W
Const_2Const*
_output_shapes
: *
dtype0*
valueB
 *дp}?2	
Const_2Я
AssignMovingAvg/sub/xConst**
_class 
loc:@AssignMovingAvg/2736277*
_output_shapes
: *
dtype0*
valueB
 *  А?2
AssignMovingAvg/sub/x░
AssignMovingAvg/subSubAssignMovingAvg/sub/x:output:0Const_2:output:0*
T0**
_class 
loc:@AssignMovingAvg/2736277*
_output_shapes
: 2
AssignMovingAvg/subФ
AssignMovingAvg/ReadVariableOpReadVariableOpassignmovingavg_2736277*
_output_shapes
: *
dtype02 
AssignMovingAvg/ReadVariableOp═
AssignMovingAvg/sub_1Sub&AssignMovingAvg/ReadVariableOp:value:0FusedBatchNormV3:batch_mean:0*
T0**
_class 
loc:@AssignMovingAvg/2736277*
_output_shapes
: 2
AssignMovingAvg/sub_1╢
AssignMovingAvg/mulMulAssignMovingAvg/sub_1:z:0AssignMovingAvg/sub:z:0*
T0**
_class 
loc:@AssignMovingAvg/2736277*
_output_shapes
: 2
AssignMovingAvg/mulГ
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOpassignmovingavg_2736277AssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp**
_class 
loc:@AssignMovingAvg/2736277*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOpе
AssignMovingAvg_1/sub/xConst*,
_class"
 loc:@AssignMovingAvg_1/2736284*
_output_shapes
: *
dtype0*
valueB
 *  А?2
AssignMovingAvg_1/sub/x╕
AssignMovingAvg_1/subSub AssignMovingAvg_1/sub/x:output:0Const_2:output:0*
T0*,
_class"
 loc:@AssignMovingAvg_1/2736284*
_output_shapes
: 2
AssignMovingAvg_1/subЪ
 AssignMovingAvg_1/ReadVariableOpReadVariableOpassignmovingavg_1_2736284*
_output_shapes
: *
dtype02"
 AssignMovingAvg_1/ReadVariableOp┘
AssignMovingAvg_1/sub_1Sub(AssignMovingAvg_1/ReadVariableOp:value:0!FusedBatchNormV3:batch_variance:0*
T0*,
_class"
 loc:@AssignMovingAvg_1/2736284*
_output_shapes
: 2
AssignMovingAvg_1/sub_1└
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub_1:z:0AssignMovingAvg_1/sub:z:0*
T0*,
_class"
 loc:@AssignMovingAvg_1/2736284*
_output_shapes
: 2
AssignMovingAvg_1/mulП
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOpassignmovingavg_1_2736284AssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*,
_class"
 loc:@AssignMovingAvg_1/2736284*
_output_shapes
 *
dtype02'
%AssignMovingAvg_1/AssignSubVariableOpж
IdentityIdentityFusedBatchNormV3:y:0$^AssignMovingAvg/AssignSubVariableOp^AssignMovingAvg/ReadVariableOp&^AssignMovingAvg_1/AssignSubVariableOp!^AssignMovingAvg_1/ReadVariableOp^ReadVariableOp^ReadVariableOp_1*
T0*/
_output_shapes
:         @@ 2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:         @@ ::::2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp2D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp2 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:& "
 
_user_specified_nameinputs
┌
є
*__inference_b_norm_3_layer_call_fn_2738159

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4
identityИвStatefulPartitionedCallч
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4*
Tin	
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*B
_output_shapes0
.:,                           А*-
config_proto

GPU

CPU2*0J 8*N
fIRG
E__inference_b_norm_3_layer_call_and_return_conditional_losses_27359352
StatefulPartitionedCallй
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*B
_output_shapes0
.:,                           А2

Identity"
identityIdentity:output:0*Q
_input_shapes@
>:,                           А::::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs
Ч$
Т
E__inference_b_norm_3_layer_call_and_return_conditional_losses_2736484

inputs
readvariableop_resource
readvariableop_1_resource
assignmovingavg_2736469
assignmovingavg_1_2736476
identityИв#AssignMovingAvg/AssignSubVariableOpвAssignMovingAvg/ReadVariableOpв%AssignMovingAvg_1/AssignSubVariableOpв AssignMovingAvg_1/ReadVariableOpвReadVariableOpвReadVariableOp_1^
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

LogicalAndu
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:А*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:А*
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
Const_1К
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0Const:output:0Const_1:output:0*
T0*
U0*P
_output_shapes>
<:         А:А:А:А:А:*
epsilon%oГ:2
FusedBatchNormV3W
Const_2Const*
_output_shapes
: *
dtype0*
valueB
 *дp}?2	
Const_2Я
AssignMovingAvg/sub/xConst**
_class 
loc:@AssignMovingAvg/2736469*
_output_shapes
: *
dtype0*
valueB
 *  А?2
AssignMovingAvg/sub/x░
AssignMovingAvg/subSubAssignMovingAvg/sub/x:output:0Const_2:output:0*
T0**
_class 
loc:@AssignMovingAvg/2736469*
_output_shapes
: 2
AssignMovingAvg/subХ
AssignMovingAvg/ReadVariableOpReadVariableOpassignmovingavg_2736469*
_output_shapes	
:А*
dtype02 
AssignMovingAvg/ReadVariableOp╬
AssignMovingAvg/sub_1Sub&AssignMovingAvg/ReadVariableOp:value:0FusedBatchNormV3:batch_mean:0*
T0**
_class 
loc:@AssignMovingAvg/2736469*
_output_shapes	
:А2
AssignMovingAvg/sub_1╖
AssignMovingAvg/mulMulAssignMovingAvg/sub_1:z:0AssignMovingAvg/sub:z:0*
T0**
_class 
loc:@AssignMovingAvg/2736469*
_output_shapes	
:А2
AssignMovingAvg/mulГ
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOpassignmovingavg_2736469AssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp**
_class 
loc:@AssignMovingAvg/2736469*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOpе
AssignMovingAvg_1/sub/xConst*,
_class"
 loc:@AssignMovingAvg_1/2736476*
_output_shapes
: *
dtype0*
valueB
 *  А?2
AssignMovingAvg_1/sub/x╕
AssignMovingAvg_1/subSub AssignMovingAvg_1/sub/x:output:0Const_2:output:0*
T0*,
_class"
 loc:@AssignMovingAvg_1/2736476*
_output_shapes
: 2
AssignMovingAvg_1/subЫ
 AssignMovingAvg_1/ReadVariableOpReadVariableOpassignmovingavg_1_2736476*
_output_shapes	
:А*
dtype02"
 AssignMovingAvg_1/ReadVariableOp┌
AssignMovingAvg_1/sub_1Sub(AssignMovingAvg_1/ReadVariableOp:value:0!FusedBatchNormV3:batch_variance:0*
T0*,
_class"
 loc:@AssignMovingAvg_1/2736476*
_output_shapes	
:А2
AssignMovingAvg_1/sub_1┴
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub_1:z:0AssignMovingAvg_1/sub:z:0*
T0*,
_class"
 loc:@AssignMovingAvg_1/2736476*
_output_shapes	
:А2
AssignMovingAvg_1/mulП
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOpassignmovingavg_1_2736476AssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*,
_class"
 loc:@AssignMovingAvg_1/2736476*
_output_shapes
 *
dtype02'
%AssignMovingAvg_1/AssignSubVariableOpз
IdentityIdentityFusedBatchNormV3:y:0$^AssignMovingAvg/AssignSubVariableOp^AssignMovingAvg/ReadVariableOp&^AssignMovingAvg_1/AssignSubVariableOp!^AssignMovingAvg_1/ReadVariableOp^ReadVariableOp^ReadVariableOp_1*
T0*0
_output_shapes
:         А2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:         А::::2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp2D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp2 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:& "
 
_user_specified_nameinputs
╞
№	
,__inference_awsome_net_layer_call_fn_2737528

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
identityИвStatefulPartitionedCallВ
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4statefulpartitionedcall_args_5statefulpartitionedcall_args_6statefulpartitionedcall_args_7statefulpartitionedcall_args_8statefulpartitionedcall_args_9statefulpartitionedcall_args_10statefulpartitionedcall_args_11statefulpartitionedcall_args_12statefulpartitionedcall_args_13statefulpartitionedcall_args_14statefulpartitionedcall_args_15statefulpartitionedcall_args_16statefulpartitionedcall_args_17statefulpartitionedcall_args_18statefulpartitionedcall_args_19statefulpartitionedcall_args_20statefulpartitionedcall_args_21statefulpartitionedcall_args_22statefulpartitionedcall_args_23statefulpartitionedcall_args_24statefulpartitionedcall_args_25statefulpartitionedcall_args_26statefulpartitionedcall_args_27statefulpartitionedcall_args_28statefulpartitionedcall_args_29statefulpartitionedcall_args_30statefulpartitionedcall_args_31statefulpartitionedcall_args_32*,
Tin%
#2!*
Tout
2*,
_gradient_op_typePartitionedCallUnused*(
_output_shapes
:         ╚*-
config_proto

GPU

CPU2*0J 8*P
fKRI
G__inference_awsome_net_layer_call_and_return_conditional_losses_27369352
StatefulPartitionedCallП
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:         ╚2

Identity"
identityIdentity:output:0*▓
_input_shapesа
Э:         АА::::::::::::::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs
Ў

▄
C__inference_conv_3_layer_call_and_return_conditional_losses_2735802

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identityИвBiasAdd/ReadVariableOpвConv2D/ReadVariableOpo
dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      2
dilation_rateЦ
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*'
_output_shapes
:@А*
dtype02
Conv2D/ReadVariableOp╢
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,                           А*
paddingSAME*
strides
2
Conv2DН
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02
BiasAdd/ReadVariableOpЫ
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,                           А2	
BiasAdd░
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*B
_output_shapes0
.:,                           А2

Identity"
identityIdentity:output:0*H
_input_shapes7
5:+                           @::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:& "
 
_user_specified_nameinputs
┐
ш
E__inference_b_norm_1_layer_call_and_return_conditional_losses_2737801

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityИвFusedBatchNormV3/ReadVariableOpв!FusedBatchNormV3/ReadVariableOp_1вReadVariableOpвReadVariableOp_1^
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
ReadVariableOp_1з
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02!
FusedBatchNormV3/ReadVariableOpн
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02#
!FusedBatchNormV3/ReadVariableOp_1╩
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:         @@ : : : : :*
epsilon%oГ:*
is_training( 2
FusedBatchNormV3S
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *дp}?2
Const┌
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*/
_output_shapes
:         @@ 2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:         @@ ::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:& "
 
_user_specified_nameinputs
Ь
╝
;__inference_output_class_distribution_layer_call_fn_2738476

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2
identityИвStatefulPartitionedCallЬ
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*(
_output_shapes
:         ╚*-
config_proto

GPU

CPU2*0J 8*_
fZRX
V__inference_output_class_distribution_layer_call_and_return_conditional_losses_27367022
StatefulPartitionedCallП
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:         ╚2

Identity"
identityIdentity:output:0*/
_input_shapes
:         А::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs
├
й
(__inference_conv_3_layer_call_fn_2735810

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2
identityИвStatefulPartitionedCallг
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*B
_output_shapes0
.:,                           А*-
config_proto

GPU

CPU2*0J 8*L
fGRE
C__inference_conv_3_layer_call_and_return_conditional_losses_27358022
StatefulPartitionedCallй
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*B
_output_shapes0
.:,                           А2

Identity"
identityIdentity:output:0*H
_input_shapes7
5:+                           @::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs
И$
Т
E__inference_b_norm_2_layer_call_and_return_conditional_losses_2738023

inputs
readvariableop_resource
readvariableop_1_resource
assignmovingavg_2738008
assignmovingavg_1_2738015
identityИв#AssignMovingAvg/AssignSubVariableOpвAssignMovingAvg/ReadVariableOpв%AssignMovingAvg_1/AssignSubVariableOpв AssignMovingAvg_1/ReadVariableOpвReadVariableOpвReadVariableOp_1^
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
Const_1Е
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0Const:output:0Const_1:output:0*
T0*
U0*K
_output_shapes9
7:           @:@:@:@:@:*
epsilon%oГ:2
FusedBatchNormV3W
Const_2Const*
_output_shapes
: *
dtype0*
valueB
 *дp}?2	
Const_2Я
AssignMovingAvg/sub/xConst**
_class 
loc:@AssignMovingAvg/2738008*
_output_shapes
: *
dtype0*
valueB
 *  А?2
AssignMovingAvg/sub/x░
AssignMovingAvg/subSubAssignMovingAvg/sub/x:output:0Const_2:output:0*
T0**
_class 
loc:@AssignMovingAvg/2738008*
_output_shapes
: 2
AssignMovingAvg/subФ
AssignMovingAvg/ReadVariableOpReadVariableOpassignmovingavg_2738008*
_output_shapes
:@*
dtype02 
AssignMovingAvg/ReadVariableOp═
AssignMovingAvg/sub_1Sub&AssignMovingAvg/ReadVariableOp:value:0FusedBatchNormV3:batch_mean:0*
T0**
_class 
loc:@AssignMovingAvg/2738008*
_output_shapes
:@2
AssignMovingAvg/sub_1╢
AssignMovingAvg/mulMulAssignMovingAvg/sub_1:z:0AssignMovingAvg/sub:z:0*
T0**
_class 
loc:@AssignMovingAvg/2738008*
_output_shapes
:@2
AssignMovingAvg/mulГ
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOpassignmovingavg_2738008AssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp**
_class 
loc:@AssignMovingAvg/2738008*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOpе
AssignMovingAvg_1/sub/xConst*,
_class"
 loc:@AssignMovingAvg_1/2738015*
_output_shapes
: *
dtype0*
valueB
 *  А?2
AssignMovingAvg_1/sub/x╕
AssignMovingAvg_1/subSub AssignMovingAvg_1/sub/x:output:0Const_2:output:0*
T0*,
_class"
 loc:@AssignMovingAvg_1/2738015*
_output_shapes
: 2
AssignMovingAvg_1/subЪ
 AssignMovingAvg_1/ReadVariableOpReadVariableOpassignmovingavg_1_2738015*
_output_shapes
:@*
dtype02"
 AssignMovingAvg_1/ReadVariableOp┘
AssignMovingAvg_1/sub_1Sub(AssignMovingAvg_1/ReadVariableOp:value:0!FusedBatchNormV3:batch_variance:0*
T0*,
_class"
 loc:@AssignMovingAvg_1/2738015*
_output_shapes
:@2
AssignMovingAvg_1/sub_1└
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub_1:z:0AssignMovingAvg_1/sub:z:0*
T0*,
_class"
 loc:@AssignMovingAvg_1/2738015*
_output_shapes
:@2
AssignMovingAvg_1/mulП
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOpassignmovingavg_1_2738015AssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*,
_class"
 loc:@AssignMovingAvg_1/2738015*
_output_shapes
 *
dtype02'
%AssignMovingAvg_1/AssignSubVariableOpж
IdentityIdentityFusedBatchNormV3:y:0$^AssignMovingAvg/AssignSubVariableOp^AssignMovingAvg/ReadVariableOp&^AssignMovingAvg_1/AssignSubVariableOp!^AssignMovingAvg_1/ReadVariableOp^ReadVariableOp^ReadVariableOp_1*
T0*/
_output_shapes
:           @2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:           @::::2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp2D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp2 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:& "
 
_user_specified_nameinputs
┌
є
*__inference_b_norm_4_layer_call_fn_2738394

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4
identityИвStatefulPartitionedCallч
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4*
Tin	
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*B
_output_shapes0
.:,                           А*-
config_proto

GPU

CPU2*0J 8*N
fIRG
E__inference_b_norm_4_layer_call_and_return_conditional_losses_27360682
StatefulPartitionedCallй
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*B
_output_shapes0
.:,                           А2

Identity"
identityIdentity:output:0*Q
_input_shapes@
>:,                           А::::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs
╓
∙
V__inference_input_batch_normalization_layer_call_and_return_conditional_losses_2737596

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityИвFusedBatchNormV3/ReadVariableOpв!FusedBatchNormV3/ReadVariableOp_1вReadVariableOpвReadVariableOp_1^
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
ReadVariableOp_1з
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02!
FusedBatchNormV3/ReadVariableOpн
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02#
!FusedBatchNormV3/ReadVariableOp_1╠
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*M
_output_shapes;
9:         АА : : : : :*
epsilon%oГ:*
is_training( 2
FusedBatchNormV3S
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *дp}?2
Const▄
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*1
_output_shapes
:         АА 2

Identity"
identityIdentity:output:0*@
_input_shapes/
-:         АА ::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:& "
 
_user_specified_nameinputs
д
e
G__inference_dropout_29_layer_call_and_return_conditional_losses_2736679

inputs

identity_1[
IdentityIdentityinputs*
T0*(
_output_shapes
:         А2

Identityj

Identity_1IdentityIdentity:output:0*
T0*(
_output_shapes
:         А2

Identity_1"!

identity_1Identity_1:output:0*'
_input_shapes
:         А:& "
 
_user_specified_nameinputs
╖
i
M__inference_max_pooling2d_59_layer_call_and_return_conditional_losses_2736112

inputs
identityн
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4                                    *
ksize
*
paddingVALID*
strides
2	
MaxPoolЗ
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4                                    2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4                                    :& "
 
_user_specified_nameinputs
ь
D
(__inference_RELU_1_layer_call_fn_2737903

inputs
identity╢
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*/
_output_shapes
:         @@ *-
config_proto

GPU

CPU2*0J 8*L
fGRE
C__inference_RELU_1_layer_call_and_return_conditional_losses_27363432
PartitionedCallt
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:         @@ 2

Identity"
identityIdentity:output:0*.
_input_shapes
:         @@ :& "
 
_user_specified_nameinputs
°

▄
C__inference_conv_4_layer_call_and_return_conditional_losses_2735966

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identityИвBiasAdd/ReadVariableOpвConv2D/ReadVariableOpo
dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      2
dilation_rateЧ
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:АА*
dtype02
Conv2D/ReadVariableOp╢
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,                           А*
paddingSAME*
strides
2
Conv2DН
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02
BiasAdd/ReadVariableOpЫ
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,                           А2	
BiasAdd░
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*B
_output_shapes0
.:,                           А2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:,                           А::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:& "
 
_user_specified_nameinputs
╖
i
M__inference_max_pooling2d_56_layer_call_and_return_conditional_losses_2735620

inputs
identityн
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4                                    *
ksize
*
paddingVALID*
strides
2	
MaxPoolЗ
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4                                    2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4                                    :& "
 
_user_specified_nameinputs
м
f
G__inference_dropout_29_layer_call_and_return_conditional_losses_2736674

inputs
identityИa
dropout/rateConst*
_output_shapes
: *
dtype0*
valueB
 *  А>2
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
 *  А?2
dropout/random_uniform/max╡
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*(
_output_shapes
:         А*
dtype02&
$dropout/random_uniform/RandomUniformк
dropout/random_uniform/subSub#dropout/random_uniform/max:output:0#dropout/random_uniform/min:output:0*
T0*
_output_shapes
: 2
dropout/random_uniform/sub┴
dropout/random_uniform/mulMul-dropout/random_uniform/RandomUniform:output:0dropout/random_uniform/sub:z:0*
T0*(
_output_shapes
:         А2
dropout/random_uniform/mulп
dropout/random_uniformAdddropout/random_uniform/mul:z:0#dropout/random_uniform/min:output:0*
T0*(
_output_shapes
:         А2
dropout/random_uniformc
dropout/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  А?2
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
 *  А?2
dropout/truediv/x{
dropout/truedivRealDivdropout/truediv/x:output:0dropout/sub:z:0*
T0*
_output_shapes
: 2
dropout/truedivв
dropout/GreaterEqualGreaterEqualdropout/random_uniform:z:0dropout/rate:output:0*
T0*(
_output_shapes
:         А2
dropout/GreaterEqualq
dropout/mulMulinputsdropout/truediv:z:0*
T0*(
_output_shapes
:         А2
dropout/mulА
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:         А2
dropout/Cast{
dropout/mul_1Muldropout/mul:z:0dropout/Cast:y:0*
T0*(
_output_shapes
:         А2
dropout/mul_1f
IdentityIdentitydropout/mul_1:z:0*
T0*(
_output_shapes
:         А2

Identity"
identityIdentity:output:0*'
_input_shapes
:         А:& "
 
_user_specified_nameinputs
г
_
C__inference_RELU_4_layer_call_and_return_conditional_losses_2736631

inputs
identityW
ReluReluinputs*
T0*0
_output_shapes
:         А2
Reluo
IdentityIdentityRelu:activations:0*
T0*0
_output_shapes
:         А2

Identity"
identityIdentity:output:0*/
_input_shapes
:         А:& "
 
_user_specified_nameinputs
═$
Т
E__inference_b_norm_3_layer_call_and_return_conditional_losses_2738119

inputs
readvariableop_resource
readvariableop_1_resource
assignmovingavg_2738104
assignmovingavg_1_2738111
identityИв#AssignMovingAvg/AssignSubVariableOpвAssignMovingAvg/ReadVariableOpв%AssignMovingAvg_1/AssignSubVariableOpв AssignMovingAvg_1/ReadVariableOpвReadVariableOpвReadVariableOp_1^
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

LogicalAndu
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:А*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:А*
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
Const_1Ь
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0Const:output:0Const_1:output:0*
T0*
U0*b
_output_shapesP
N:,                           А:А:А:А:А:*
epsilon%oГ:2
FusedBatchNormV3W
Const_2Const*
_output_shapes
: *
dtype0*
valueB
 *дp}?2	
Const_2Я
AssignMovingAvg/sub/xConst**
_class 
loc:@AssignMovingAvg/2738104*
_output_shapes
: *
dtype0*
valueB
 *  А?2
AssignMovingAvg/sub/x░
AssignMovingAvg/subSubAssignMovingAvg/sub/x:output:0Const_2:output:0*
T0**
_class 
loc:@AssignMovingAvg/2738104*
_output_shapes
: 2
AssignMovingAvg/subХ
AssignMovingAvg/ReadVariableOpReadVariableOpassignmovingavg_2738104*
_output_shapes	
:А*
dtype02 
AssignMovingAvg/ReadVariableOp╬
AssignMovingAvg/sub_1Sub&AssignMovingAvg/ReadVariableOp:value:0FusedBatchNormV3:batch_mean:0*
T0**
_class 
loc:@AssignMovingAvg/2738104*
_output_shapes	
:А2
AssignMovingAvg/sub_1╖
AssignMovingAvg/mulMulAssignMovingAvg/sub_1:z:0AssignMovingAvg/sub:z:0*
T0**
_class 
loc:@AssignMovingAvg/2738104*
_output_shapes	
:А2
AssignMovingAvg/mulГ
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOpassignmovingavg_2738104AssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp**
_class 
loc:@AssignMovingAvg/2738104*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOpе
AssignMovingAvg_1/sub/xConst*,
_class"
 loc:@AssignMovingAvg_1/2738111*
_output_shapes
: *
dtype0*
valueB
 *  А?2
AssignMovingAvg_1/sub/x╕
AssignMovingAvg_1/subSub AssignMovingAvg_1/sub/x:output:0Const_2:output:0*
T0*,
_class"
 loc:@AssignMovingAvg_1/2738111*
_output_shapes
: 2
AssignMovingAvg_1/subЫ
 AssignMovingAvg_1/ReadVariableOpReadVariableOpassignmovingavg_1_2738111*
_output_shapes	
:А*
dtype02"
 AssignMovingAvg_1/ReadVariableOp┌
AssignMovingAvg_1/sub_1Sub(AssignMovingAvg_1/ReadVariableOp:value:0!FusedBatchNormV3:batch_variance:0*
T0*,
_class"
 loc:@AssignMovingAvg_1/2738111*
_output_shapes	
:А2
AssignMovingAvg_1/sub_1┴
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub_1:z:0AssignMovingAvg_1/sub:z:0*
T0*,
_class"
 loc:@AssignMovingAvg_1/2738111*
_output_shapes	
:А2
AssignMovingAvg_1/mulП
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOpassignmovingavg_1_2738111AssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*,
_class"
 loc:@AssignMovingAvg_1/2738111*
_output_shapes
 *
dtype02'
%AssignMovingAvg_1/AssignSubVariableOp╣
IdentityIdentityFusedBatchNormV3:y:0$^AssignMovingAvg/AssignSubVariableOp^AssignMovingAvg/ReadVariableOp&^AssignMovingAvg_1/AssignSubVariableOp!^AssignMovingAvg_1/ReadVariableOp^ReadVariableOp^ReadVariableOp_1*
T0*B
_output_shapes0
.:,                           А2

Identity"
identityIdentity:output:0*Q
_input_shapes@
>:,                           А::::2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp2D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp2 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:& "
 
_user_specified_nameinputs
ф
Ж

,__inference_awsome_net_layer_call_fn_2736970
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
identityИвStatefulPartitionedCallМ
StatefulPartitionedCallStatefulPartitionedCallinput_conv_inputstatefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4statefulpartitionedcall_args_5statefulpartitionedcall_args_6statefulpartitionedcall_args_7statefulpartitionedcall_args_8statefulpartitionedcall_args_9statefulpartitionedcall_args_10statefulpartitionedcall_args_11statefulpartitionedcall_args_12statefulpartitionedcall_args_13statefulpartitionedcall_args_14statefulpartitionedcall_args_15statefulpartitionedcall_args_16statefulpartitionedcall_args_17statefulpartitionedcall_args_18statefulpartitionedcall_args_19statefulpartitionedcall_args_20statefulpartitionedcall_args_21statefulpartitionedcall_args_22statefulpartitionedcall_args_23statefulpartitionedcall_args_24statefulpartitionedcall_args_25statefulpartitionedcall_args_26statefulpartitionedcall_args_27statefulpartitionedcall_args_28statefulpartitionedcall_args_29statefulpartitionedcall_args_30statefulpartitionedcall_args_31statefulpartitionedcall_args_32*,
Tin%
#2!*
Tout
2*,
_gradient_op_typePartitionedCallUnused*(
_output_shapes
:         ╚*-
config_proto

GPU

CPU2*0J 8*P
fKRI
G__inference_awsome_net_layer_call_and_return_conditional_losses_27369352
StatefulPartitionedCallП
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:         ╚2

Identity"
identityIdentity:output:0*▓
_input_shapesа
Э:         АА::::::::::::::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:0 ,
*
_user_specified_nameinput_conv_input
╔
Д
;__inference_input_batch_normalization_layer_call_fn_2737605

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4
identityИвStatefulPartitionedCallч
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4*
Tin	
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*1
_output_shapes
:         АА *-
config_proto

GPU

CPU2*0J 8*_
fZRX
V__inference_input_batch_normalization_layer_call_and_return_conditional_losses_27361582
StatefulPartitionedCallШ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*1
_output_shapes
:         АА 2

Identity"
identityIdentity:output:0*@
_input_shapes/
-:         АА ::::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs
А
e
,__inference_dropout_28_layer_call_fn_2737728

inputs
identityИвStatefulPartitionedCall╥
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*/
_output_shapes
:         @@ *-
config_proto

GPU

CPU2*0J 8*P
fKRI
G__inference_dropout_28_layer_call_and_return_conditional_losses_27362382
StatefulPartitionedCallЦ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:         @@ 2

Identity"
identityIdentity:output:0*.
_input_shapes
:         @@ 22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs
Я$
г
V__inference_input_batch_normalization_layer_call_and_return_conditional_losses_2737574

inputs
readvariableop_resource
readvariableop_1_resource
assignmovingavg_2737559
assignmovingavg_1_2737566
identityИв#AssignMovingAvg/AssignSubVariableOpвAssignMovingAvg/ReadVariableOpв%AssignMovingAvg_1/AssignSubVariableOpв AssignMovingAvg_1/ReadVariableOpвReadVariableOpвReadVariableOp_1^
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
Const_1З
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0Const:output:0Const_1:output:0*
T0*
U0*M
_output_shapes;
9:         АА : : : : :*
epsilon%oГ:2
FusedBatchNormV3W
Const_2Const*
_output_shapes
: *
dtype0*
valueB
 *дp}?2	
Const_2Я
AssignMovingAvg/sub/xConst**
_class 
loc:@AssignMovingAvg/2737559*
_output_shapes
: *
dtype0*
valueB
 *  А?2
AssignMovingAvg/sub/x░
AssignMovingAvg/subSubAssignMovingAvg/sub/x:output:0Const_2:output:0*
T0**
_class 
loc:@AssignMovingAvg/2737559*
_output_shapes
: 2
AssignMovingAvg/subФ
AssignMovingAvg/ReadVariableOpReadVariableOpassignmovingavg_2737559*
_output_shapes
: *
dtype02 
AssignMovingAvg/ReadVariableOp═
AssignMovingAvg/sub_1Sub&AssignMovingAvg/ReadVariableOp:value:0FusedBatchNormV3:batch_mean:0*
T0**
_class 
loc:@AssignMovingAvg/2737559*
_output_shapes
: 2
AssignMovingAvg/sub_1╢
AssignMovingAvg/mulMulAssignMovingAvg/sub_1:z:0AssignMovingAvg/sub:z:0*
T0**
_class 
loc:@AssignMovingAvg/2737559*
_output_shapes
: 2
AssignMovingAvg/mulГ
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOpassignmovingavg_2737559AssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp**
_class 
loc:@AssignMovingAvg/2737559*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOpе
AssignMovingAvg_1/sub/xConst*,
_class"
 loc:@AssignMovingAvg_1/2737566*
_output_shapes
: *
dtype0*
valueB
 *  А?2
AssignMovingAvg_1/sub/x╕
AssignMovingAvg_1/subSub AssignMovingAvg_1/sub/x:output:0Const_2:output:0*
T0*,
_class"
 loc:@AssignMovingAvg_1/2737566*
_output_shapes
: 2
AssignMovingAvg_1/subЪ
 AssignMovingAvg_1/ReadVariableOpReadVariableOpassignmovingavg_1_2737566*
_output_shapes
: *
dtype02"
 AssignMovingAvg_1/ReadVariableOp┘
AssignMovingAvg_1/sub_1Sub(AssignMovingAvg_1/ReadVariableOp:value:0!FusedBatchNormV3:batch_variance:0*
T0*,
_class"
 loc:@AssignMovingAvg_1/2737566*
_output_shapes
: 2
AssignMovingAvg_1/sub_1└
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub_1:z:0AssignMovingAvg_1/sub:z:0*
T0*,
_class"
 loc:@AssignMovingAvg_1/2737566*
_output_shapes
: 2
AssignMovingAvg_1/mulП
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOpassignmovingavg_1_2737566AssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*,
_class"
 loc:@AssignMovingAvg_1/2737566*
_output_shapes
 *
dtype02'
%AssignMovingAvg_1/AssignSubVariableOpи
IdentityIdentityFusedBatchNormV3:y:0$^AssignMovingAvg/AssignSubVariableOp^AssignMovingAvg/ReadVariableOp&^AssignMovingAvg_1/AssignSubVariableOp!^AssignMovingAvg_1/ReadVariableOp^ReadVariableOp^ReadVariableOp_1*
T0*1
_output_shapes
:         АА 2

Identity"
identityIdentity:output:0*@
_input_shapes/
-:         АА ::::2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp2D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp2 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:& "
 
_user_specified_nameinputs
ї
ш
E__inference_b_norm_2_layer_call_and_return_conditional_losses_2737971

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityИвFusedBatchNormV3/ReadVariableOpв!FusedBatchNormV3/ReadVariableOp_1вReadVariableOpвReadVariableOp_1^
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
ReadVariableOp_1з
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOpн
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1▄
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+                           @:@:@:@:@:*
epsilon%oГ:*
is_training( 2
FusedBatchNormV3S
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *дp}?2
Constь
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*A
_output_shapes/
-:+                           @2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+                           @::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:& "
 
_user_specified_nameinputs
Ч$
Т
E__inference_b_norm_4_layer_call_and_return_conditional_losses_2738289

inputs
readvariableop_resource
readvariableop_1_resource
assignmovingavg_2738274
assignmovingavg_1_2738281
identityИв#AssignMovingAvg/AssignSubVariableOpвAssignMovingAvg/ReadVariableOpв%AssignMovingAvg_1/AssignSubVariableOpв AssignMovingAvg_1/ReadVariableOpвReadVariableOpвReadVariableOp_1^
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

LogicalAndu
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:А*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:А*
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
Const_1К
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0Const:output:0Const_1:output:0*
T0*
U0*P
_output_shapes>
<:         А:А:А:А:А:*
epsilon%oГ:2
FusedBatchNormV3W
Const_2Const*
_output_shapes
: *
dtype0*
valueB
 *дp}?2	
Const_2Я
AssignMovingAvg/sub/xConst**
_class 
loc:@AssignMovingAvg/2738274*
_output_shapes
: *
dtype0*
valueB
 *  А?2
AssignMovingAvg/sub/x░
AssignMovingAvg/subSubAssignMovingAvg/sub/x:output:0Const_2:output:0*
T0**
_class 
loc:@AssignMovingAvg/2738274*
_output_shapes
: 2
AssignMovingAvg/subХ
AssignMovingAvg/ReadVariableOpReadVariableOpassignmovingavg_2738274*
_output_shapes	
:А*
dtype02 
AssignMovingAvg/ReadVariableOp╬
AssignMovingAvg/sub_1Sub&AssignMovingAvg/ReadVariableOp:value:0FusedBatchNormV3:batch_mean:0*
T0**
_class 
loc:@AssignMovingAvg/2738274*
_output_shapes	
:А2
AssignMovingAvg/sub_1╖
AssignMovingAvg/mulMulAssignMovingAvg/sub_1:z:0AssignMovingAvg/sub:z:0*
T0**
_class 
loc:@AssignMovingAvg/2738274*
_output_shapes	
:А2
AssignMovingAvg/mulГ
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOpassignmovingavg_2738274AssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp**
_class 
loc:@AssignMovingAvg/2738274*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOpе
AssignMovingAvg_1/sub/xConst*,
_class"
 loc:@AssignMovingAvg_1/2738281*
_output_shapes
: *
dtype0*
valueB
 *  А?2
AssignMovingAvg_1/sub/x╕
AssignMovingAvg_1/subSub AssignMovingAvg_1/sub/x:output:0Const_2:output:0*
T0*,
_class"
 loc:@AssignMovingAvg_1/2738281*
_output_shapes
: 2
AssignMovingAvg_1/subЫ
 AssignMovingAvg_1/ReadVariableOpReadVariableOpassignmovingavg_1_2738281*
_output_shapes	
:А*
dtype02"
 AssignMovingAvg_1/ReadVariableOp┌
AssignMovingAvg_1/sub_1Sub(AssignMovingAvg_1/ReadVariableOp:value:0!FusedBatchNormV3:batch_variance:0*
T0*,
_class"
 loc:@AssignMovingAvg_1/2738281*
_output_shapes	
:А2
AssignMovingAvg_1/sub_1┴
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub_1:z:0AssignMovingAvg_1/sub:z:0*
T0*,
_class"
 loc:@AssignMovingAvg_1/2738281*
_output_shapes	
:А2
AssignMovingAvg_1/mulП
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOpassignmovingavg_1_2738281AssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*,
_class"
 loc:@AssignMovingAvg_1/2738281*
_output_shapes
 *
dtype02'
%AssignMovingAvg_1/AssignSubVariableOpз
IdentityIdentityFusedBatchNormV3:y:0$^AssignMovingAvg/AssignSubVariableOp^AssignMovingAvg/ReadVariableOp&^AssignMovingAvg_1/AssignSubVariableOp!^AssignMovingAvg_1/ReadVariableOp^ReadVariableOp^ReadVariableOp_1*
T0*0
_output_shapes
:         А2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:         А::::2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp2D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp2 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:& "
 
_user_specified_nameinputs
г
_
C__inference_RELU_4_layer_call_and_return_conditional_losses_2738408

inputs
identityW
ReluReluinputs*
T0*0
_output_shapes
:         А2
Reluo
IdentityIdentityRelu:activations:0*
T0*0
_output_shapes
:         А2

Identity"
identityIdentity:output:0*/
_input_shapes
:         А:& "
 
_user_specified_nameinputs
я
D
(__inference_RELU_3_layer_call_fn_2738243

inputs
identity╖
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*0
_output_shapes
:         А*-
config_proto

GPU

CPU2*0J 8*L
fGRE
C__inference_RELU_3_layer_call_and_return_conditional_losses_27365352
PartitionedCallu
IdentityIdentityPartitionedCall:output:0*
T0*0
_output_shapes
:         А2

Identity"
identityIdentity:output:0*/
_input_shapes
:         А:& "
 
_user_specified_nameinputs
╖
i
M__inference_max_pooling2d_58_layer_call_and_return_conditional_losses_2735948

inputs
identityн
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4                                    *
ksize
*
paddingVALID*
strides
2	
MaxPoolЗ
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4                                    2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4                                    :& "
 
_user_specified_nameinputs
╛$
Т
E__inference_b_norm_2_layer_call_and_return_conditional_losses_2735740

inputs
readvariableop_resource
readvariableop_1_resource
assignmovingavg_2735725
assignmovingavg_1_2735732
identityИв#AssignMovingAvg/AssignSubVariableOpвAssignMovingAvg/ReadVariableOpв%AssignMovingAvg_1/AssignSubVariableOpв AssignMovingAvg_1/ReadVariableOpвReadVariableOpвReadVariableOp_1^
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
Const_1Ч
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0Const:output:0Const_1:output:0*
T0*
U0*]
_output_shapesK
I:+                           @:@:@:@:@:*
epsilon%oГ:2
FusedBatchNormV3W
Const_2Const*
_output_shapes
: *
dtype0*
valueB
 *дp}?2	
Const_2Я
AssignMovingAvg/sub/xConst**
_class 
loc:@AssignMovingAvg/2735725*
_output_shapes
: *
dtype0*
valueB
 *  А?2
AssignMovingAvg/sub/x░
AssignMovingAvg/subSubAssignMovingAvg/sub/x:output:0Const_2:output:0*
T0**
_class 
loc:@AssignMovingAvg/2735725*
_output_shapes
: 2
AssignMovingAvg/subФ
AssignMovingAvg/ReadVariableOpReadVariableOpassignmovingavg_2735725*
_output_shapes
:@*
dtype02 
AssignMovingAvg/ReadVariableOp═
AssignMovingAvg/sub_1Sub&AssignMovingAvg/ReadVariableOp:value:0FusedBatchNormV3:batch_mean:0*
T0**
_class 
loc:@AssignMovingAvg/2735725*
_output_shapes
:@2
AssignMovingAvg/sub_1╢
AssignMovingAvg/mulMulAssignMovingAvg/sub_1:z:0AssignMovingAvg/sub:z:0*
T0**
_class 
loc:@AssignMovingAvg/2735725*
_output_shapes
:@2
AssignMovingAvg/mulГ
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOpassignmovingavg_2735725AssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp**
_class 
loc:@AssignMovingAvg/2735725*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOpе
AssignMovingAvg_1/sub/xConst*,
_class"
 loc:@AssignMovingAvg_1/2735732*
_output_shapes
: *
dtype0*
valueB
 *  А?2
AssignMovingAvg_1/sub/x╕
AssignMovingAvg_1/subSub AssignMovingAvg_1/sub/x:output:0Const_2:output:0*
T0*,
_class"
 loc:@AssignMovingAvg_1/2735732*
_output_shapes
: 2
AssignMovingAvg_1/subЪ
 AssignMovingAvg_1/ReadVariableOpReadVariableOpassignmovingavg_1_2735732*
_output_shapes
:@*
dtype02"
 AssignMovingAvg_1/ReadVariableOp┘
AssignMovingAvg_1/sub_1Sub(AssignMovingAvg_1/ReadVariableOp:value:0!FusedBatchNormV3:batch_variance:0*
T0*,
_class"
 loc:@AssignMovingAvg_1/2735732*
_output_shapes
:@2
AssignMovingAvg_1/sub_1└
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub_1:z:0AssignMovingAvg_1/sub:z:0*
T0*,
_class"
 loc:@AssignMovingAvg_1/2735732*
_output_shapes
:@2
AssignMovingAvg_1/mulП
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOpassignmovingavg_1_2735732AssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*,
_class"
 loc:@AssignMovingAvg_1/2735732*
_output_shapes
 *
dtype02'
%AssignMovingAvg_1/AssignSubVariableOp╕
IdentityIdentityFusedBatchNormV3:y:0$^AssignMovingAvg/AssignSubVariableOp^AssignMovingAvg/ReadVariableOp&^AssignMovingAvg_1/AssignSubVariableOp!^AssignMovingAvg_1/ReadVariableOp^ReadVariableOp^ReadVariableOp_1*
T0*A
_output_shapes/
-:+                           @2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+                           @::::2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp2D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp2 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:& "
 
_user_specified_nameinputs
В	
я
V__inference_output_class_distribution_layer_call_and_return_conditional_losses_2738469

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identityИвBiasAdd/ReadVariableOpвMatMul/ReadVariableOpП
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
А╚*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ╚2
MatMulН
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:╚*
dtype02
BiasAdd/ReadVariableOpВ
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ╚2	
BiasAddЦ
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*(
_output_shapes
:         ╚2

Identity"
identityIdentity:output:0*/
_input_shapes
:         А::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:& "
 
_user_specified_nameinputs
к
c
G__inference_input_RELU_layer_call_and_return_conditional_losses_2736209

inputs
identityX
ReluReluinputs*
T0*1
_output_shapes
:         АА 2
Relup
IdentityIdentityRelu:activations:0*
T0*1
_output_shapes
:         АА 2

Identity"
identityIdentity:output:0*0
_input_shapes
:         АА :& "
 
_user_specified_nameinputs
╩
ш
E__inference_b_norm_3_layer_call_and_return_conditional_losses_2738215

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityИвFusedBatchNormV3/ReadVariableOpв!FusedBatchNormV3/ReadVariableOp_1вReadVariableOpвReadVariableOp_1^
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

LogicalAndu
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:А*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:А*
dtype02
ReadVariableOp_1и
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:А*
dtype02!
FusedBatchNormV3/ReadVariableOpо
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:А*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1╧
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:         А:А:А:А:А:*
epsilon%oГ:*
is_training( 2
FusedBatchNormV3S
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *дp}?2
Const█
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*0
_output_shapes
:         А2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:         А::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:& "
 
_user_specified_nameinputs
┌
є
*__inference_b_norm_4_layer_call_fn_2738403

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4
identityИвStatefulPartitionedCallч
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4*
Tin	
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*B
_output_shapes0
.:,                           А*-
config_proto

GPU

CPU2*0J 8*N
fIRG
E__inference_b_norm_4_layer_call_and_return_conditional_losses_27360992
StatefulPartitionedCallй
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*B
_output_shapes0
.:,                           А2

Identity"
identityIdentity:output:0*Q
_input_shapes@
>:,                           А::::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs
ї

р
G__inference_input_conv_layer_call_and_return_conditional_losses_2735310

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identityИвBiasAdd/ReadVariableOpвConv2D/ReadVariableOpo
dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      2
dilation_rateХ
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: *
dtype02
Conv2D/ReadVariableOp╡
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+                            *
paddingSAME*
strides
2
Conv2DМ
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOpЪ
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+                            2	
BiasAddп
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*A
_output_shapes/
-:+                            2

Identity"
identityIdentity:output:0*H
_input_shapes7
5:+                           ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:& "
 
_user_specified_nameinputs
А
ш
E__inference_b_norm_4_layer_call_and_return_conditional_losses_2738385

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityИвFusedBatchNormV3/ReadVariableOpв!FusedBatchNormV3/ReadVariableOp_1вReadVariableOpвReadVariableOp_1^
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

LogicalAndu
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:А*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:А*
dtype02
ReadVariableOp_1и
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:А*
dtype02!
FusedBatchNormV3/ReadVariableOpо
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:А*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1с
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,                           А:А:А:А:А:*
epsilon%oГ:*
is_training( 2
FusedBatchNormV3S
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *дp}?2
Constэ
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*B
_output_shapes0
.:,                           А2

Identity"
identityIdentity:output:0*Q
_input_shapes@
>:,                           А::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:& "
 
_user_specified_nameinputs
ё

▄
C__inference_conv_1_layer_call_and_return_conditional_losses_2735474

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identityИвBiasAdd/ReadVariableOpвConv2D/ReadVariableOpo
dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      2
dilation_rateХ
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:  *
dtype02
Conv2D/ReadVariableOp╡
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+                            *
paddingSAME*
strides
2
Conv2DМ
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOpЪ
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+                            2	
BiasAddп
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*A
_output_shapes/
-:+                            2

Identity"
identityIdentity:output:0*H
_input_shapes7
5:+                            ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:& "
 
_user_specified_nameinputs
В	
я
V__inference_output_class_distribution_layer_call_and_return_conditional_losses_2736702

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identityИвBiasAdd/ReadVariableOpвMatMul/ReadVariableOpП
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
А╚*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ╚2
MatMulН
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:╚*
dtype02
BiasAdd/ReadVariableOpВ
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ╚2	
BiasAddЦ
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*(
_output_shapes
:         ╚2

Identity"
identityIdentity:output:0*/
_input_shapes
:         А::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:& "
 
_user_specified_nameinputs
А
ш
E__inference_b_norm_3_layer_call_and_return_conditional_losses_2735935

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityИвFusedBatchNormV3/ReadVariableOpв!FusedBatchNormV3/ReadVariableOp_1вReadVariableOpвReadVariableOp_1^
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

LogicalAndu
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:А*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:А*
dtype02
ReadVariableOp_1и
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:А*
dtype02!
FusedBatchNormV3/ReadVariableOpо
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:А*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1с
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,                           А:А:А:А:А:*
epsilon%oГ:*
is_training( 2
FusedBatchNormV3S
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *дp}?2
Constэ
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*B
_output_shapes0
.:,                           А2

Identity"
identityIdentity:output:0*Q
_input_shapes@
>:,                           А::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:& "
 
_user_specified_nameinputs
ы
e
,__inference_dropout_29_layer_call_fn_2738454

inputs
identityИвStatefulPartitionedCall╦
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*(
_output_shapes
:         А*-
config_proto

GPU

CPU2*0J 8*P
fKRI
G__inference_dropout_29_layer_call_and_return_conditional_losses_27366742
StatefulPartitionedCallП
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:         А2

Identity"
identityIdentity:output:0*'
_input_shapes
:         А22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs
╕
 	
%__inference_signature_wrapper_2737076
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
identityИвStatefulPartitionedCallч

StatefulPartitionedCallStatefulPartitionedCallinput_conv_inputstatefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4statefulpartitionedcall_args_5statefulpartitionedcall_args_6statefulpartitionedcall_args_7statefulpartitionedcall_args_8statefulpartitionedcall_args_9statefulpartitionedcall_args_10statefulpartitionedcall_args_11statefulpartitionedcall_args_12statefulpartitionedcall_args_13statefulpartitionedcall_args_14statefulpartitionedcall_args_15statefulpartitionedcall_args_16statefulpartitionedcall_args_17statefulpartitionedcall_args_18statefulpartitionedcall_args_19statefulpartitionedcall_args_20statefulpartitionedcall_args_21statefulpartitionedcall_args_22statefulpartitionedcall_args_23statefulpartitionedcall_args_24statefulpartitionedcall_args_25statefulpartitionedcall_args_26statefulpartitionedcall_args_27statefulpartitionedcall_args_28statefulpartitionedcall_args_29statefulpartitionedcall_args_30statefulpartitionedcall_args_31statefulpartitionedcall_args_32*,
Tin%
#2!*
Tout
2*,
_gradient_op_typePartitionedCallUnused*(
_output_shapes
:         ╚*-
config_proto

GPU

CPU2*0J 8*+
f&R$
"__inference__wrapped_model_27352982
StatefulPartitionedCallП
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:         ╚2

Identity"
identityIdentity:output:0*▓
_input_shapesа
Э:         АА::::::::::::::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:0 ,
*
_user_specified_nameinput_conv_input
┘╤
╨.
#__inference__traced_restore_2739010
file_prefix)
%assignvariableop_input_conv_14_kernel)
%assignvariableop_1_input_conv_14_bias9
5assignvariableop_2_input_batch_normalization_14_gamma8
4assignvariableop_3_input_batch_normalization_14_beta?
;assignvariableop_4_input_batch_normalization_14_moving_meanC
?assignvariableop_5_input_batch_normalization_14_moving_variance'
#assignvariableop_6_conv_1_14_kernel%
!assignvariableop_7_conv_1_14_bias(
$assignvariableop_8_b_norm_1_14_gamma'
#assignvariableop_9_b_norm_1_14_beta/
+assignvariableop_10_b_norm_1_14_moving_mean3
/assignvariableop_11_b_norm_1_14_moving_variance(
$assignvariableop_12_conv_2_14_kernel&
"assignvariableop_13_conv_2_14_bias)
%assignvariableop_14_b_norm_2_14_gamma(
$assignvariableop_15_b_norm_2_14_beta/
+assignvariableop_16_b_norm_2_14_moving_mean3
/assignvariableop_17_b_norm_2_14_moving_variance'
#assignvariableop_18_conv_3_9_kernel%
!assignvariableop_19_conv_3_9_bias(
$assignvariableop_20_b_norm_3_9_gamma'
#assignvariableop_21_b_norm_3_9_beta.
*assignvariableop_22_b_norm_3_9_moving_mean2
.assignvariableop_23_b_norm_3_9_moving_variance'
#assignvariableop_24_conv_4_4_kernel%
!assignvariableop_25_conv_4_4_bias(
$assignvariableop_26_b_norm_4_4_gamma'
#assignvariableop_27_b_norm_4_4_beta.
*assignvariableop_28_b_norm_4_4_moving_mean2
.assignvariableop_29_b_norm_4_4_moving_variance;
7assignvariableop_30_output_class_distribution_14_kernel9
5assignvariableop_31_output_class_distribution_14_bias!
assignvariableop_32_adam_iter#
assignvariableop_33_adam_beta_1#
assignvariableop_34_adam_beta_2"
assignvariableop_35_adam_decay*
&assignvariableop_36_adam_learning_rate
assignvariableop_37_total
assignvariableop_38_count3
/assignvariableop_39_adam_input_conv_14_kernel_m1
-assignvariableop_40_adam_input_conv_14_bias_mA
=assignvariableop_41_adam_input_batch_normalization_14_gamma_m@
<assignvariableop_42_adam_input_batch_normalization_14_beta_m/
+assignvariableop_43_adam_conv_1_14_kernel_m-
)assignvariableop_44_adam_conv_1_14_bias_m0
,assignvariableop_45_adam_b_norm_1_14_gamma_m/
+assignvariableop_46_adam_b_norm_1_14_beta_m/
+assignvariableop_47_adam_conv_2_14_kernel_m-
)assignvariableop_48_adam_conv_2_14_bias_m0
,assignvariableop_49_adam_b_norm_2_14_gamma_m/
+assignvariableop_50_adam_b_norm_2_14_beta_m.
*assignvariableop_51_adam_conv_3_9_kernel_m,
(assignvariableop_52_adam_conv_3_9_bias_m/
+assignvariableop_53_adam_b_norm_3_9_gamma_m.
*assignvariableop_54_adam_b_norm_3_9_beta_m.
*assignvariableop_55_adam_conv_4_4_kernel_m,
(assignvariableop_56_adam_conv_4_4_bias_m/
+assignvariableop_57_adam_b_norm_4_4_gamma_m.
*assignvariableop_58_adam_b_norm_4_4_beta_mB
>assignvariableop_59_adam_output_class_distribution_14_kernel_m@
<assignvariableop_60_adam_output_class_distribution_14_bias_m3
/assignvariableop_61_adam_input_conv_14_kernel_v1
-assignvariableop_62_adam_input_conv_14_bias_vA
=assignvariableop_63_adam_input_batch_normalization_14_gamma_v@
<assignvariableop_64_adam_input_batch_normalization_14_beta_v/
+assignvariableop_65_adam_conv_1_14_kernel_v-
)assignvariableop_66_adam_conv_1_14_bias_v0
,assignvariableop_67_adam_b_norm_1_14_gamma_v/
+assignvariableop_68_adam_b_norm_1_14_beta_v/
+assignvariableop_69_adam_conv_2_14_kernel_v-
)assignvariableop_70_adam_conv_2_14_bias_v0
,assignvariableop_71_adam_b_norm_2_14_gamma_v/
+assignvariableop_72_adam_b_norm_2_14_beta_v.
*assignvariableop_73_adam_conv_3_9_kernel_v,
(assignvariableop_74_adam_conv_3_9_bias_v/
+assignvariableop_75_adam_b_norm_3_9_gamma_v.
*assignvariableop_76_adam_b_norm_3_9_beta_v.
*assignvariableop_77_adam_conv_4_4_kernel_v,
(assignvariableop_78_adam_conv_4_4_bias_v/
+assignvariableop_79_adam_b_norm_4_4_gamma_v.
*assignvariableop_80_adam_b_norm_4_4_beta_vB
>assignvariableop_81_adam_output_class_distribution_14_kernel_v@
<assignvariableop_82_adam_output_class_distribution_14_bias_v
identity_84ИвAssignVariableOpвAssignVariableOp_1вAssignVariableOp_10вAssignVariableOp_11вAssignVariableOp_12вAssignVariableOp_13вAssignVariableOp_14вAssignVariableOp_15вAssignVariableOp_16вAssignVariableOp_17вAssignVariableOp_18вAssignVariableOp_19вAssignVariableOp_2вAssignVariableOp_20вAssignVariableOp_21вAssignVariableOp_22вAssignVariableOp_23вAssignVariableOp_24вAssignVariableOp_25вAssignVariableOp_26вAssignVariableOp_27вAssignVariableOp_28вAssignVariableOp_29вAssignVariableOp_3вAssignVariableOp_30вAssignVariableOp_31вAssignVariableOp_32вAssignVariableOp_33вAssignVariableOp_34вAssignVariableOp_35вAssignVariableOp_36вAssignVariableOp_37вAssignVariableOp_38вAssignVariableOp_39вAssignVariableOp_4вAssignVariableOp_40вAssignVariableOp_41вAssignVariableOp_42вAssignVariableOp_43вAssignVariableOp_44вAssignVariableOp_45вAssignVariableOp_46вAssignVariableOp_47вAssignVariableOp_48вAssignVariableOp_49вAssignVariableOp_5вAssignVariableOp_50вAssignVariableOp_51вAssignVariableOp_52вAssignVariableOp_53вAssignVariableOp_54вAssignVariableOp_55вAssignVariableOp_56вAssignVariableOp_57вAssignVariableOp_58вAssignVariableOp_59вAssignVariableOp_6вAssignVariableOp_60вAssignVariableOp_61вAssignVariableOp_62вAssignVariableOp_63вAssignVariableOp_64вAssignVariableOp_65вAssignVariableOp_66вAssignVariableOp_67вAssignVariableOp_68вAssignVariableOp_69вAssignVariableOp_7вAssignVariableOp_70вAssignVariableOp_71вAssignVariableOp_72вAssignVariableOp_73вAssignVariableOp_74вAssignVariableOp_75вAssignVariableOp_76вAssignVariableOp_77вAssignVariableOp_78вAssignVariableOp_79вAssignVariableOp_8вAssignVariableOp_80вAssignVariableOp_81вAssignVariableOp_82вAssignVariableOp_9в	RestoreV2вRestoreV2_1┼.
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:S*
dtype0*╤-
value╟-B─-SB6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-1/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-1/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-1/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-3/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-3/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-3/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-5/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-5/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-5/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-7/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-7/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-7/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-7/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-8/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-8/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-9/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-9/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-9/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-9/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-10/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-10/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-1/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-3/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-5/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-7/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-7/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-8/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-8/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-9/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-9/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-10/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-10/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-1/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-3/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-5/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-7/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-7/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-8/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-8/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-9/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-9/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-10/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-10/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE2
RestoreV2/tensor_names╖
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:S*
dtype0*╗
value▒BоSB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
RestoreV2/shape_and_slices═
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*т
_output_shapes╧
╠:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::*a
dtypesW
U2S	2
	RestoreV2X
IdentityIdentityRestoreV2:tensors:0*
T0*
_output_shapes
:2

IdentityХ
AssignVariableOpAssignVariableOp%assignvariableop_input_conv_14_kernelIdentity:output:0*
_output_shapes
 *
dtype02
AssignVariableOp\

Identity_1IdentityRestoreV2:tensors:1*
T0*
_output_shapes
:2

Identity_1Ы
AssignVariableOp_1AssignVariableOp%assignvariableop_1_input_conv_14_biasIdentity_1:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_1\

Identity_2IdentityRestoreV2:tensors:2*
T0*
_output_shapes
:2

Identity_2л
AssignVariableOp_2AssignVariableOp5assignvariableop_2_input_batch_normalization_14_gammaIdentity_2:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_2\

Identity_3IdentityRestoreV2:tensors:3*
T0*
_output_shapes
:2

Identity_3к
AssignVariableOp_3AssignVariableOp4assignvariableop_3_input_batch_normalization_14_betaIdentity_3:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_3\

Identity_4IdentityRestoreV2:tensors:4*
T0*
_output_shapes
:2

Identity_4▒
AssignVariableOp_4AssignVariableOp;assignvariableop_4_input_batch_normalization_14_moving_meanIdentity_4:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_4\

Identity_5IdentityRestoreV2:tensors:5*
T0*
_output_shapes
:2

Identity_5╡
AssignVariableOp_5AssignVariableOp?assignvariableop_5_input_batch_normalization_14_moving_varianceIdentity_5:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_5\

Identity_6IdentityRestoreV2:tensors:6*
T0*
_output_shapes
:2

Identity_6Щ
AssignVariableOp_6AssignVariableOp#assignvariableop_6_conv_1_14_kernelIdentity_6:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_6\

Identity_7IdentityRestoreV2:tensors:7*
T0*
_output_shapes
:2

Identity_7Ч
AssignVariableOp_7AssignVariableOp!assignvariableop_7_conv_1_14_biasIdentity_7:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_7\

Identity_8IdentityRestoreV2:tensors:8*
T0*
_output_shapes
:2

Identity_8Ъ
AssignVariableOp_8AssignVariableOp$assignvariableop_8_b_norm_1_14_gammaIdentity_8:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_8\

Identity_9IdentityRestoreV2:tensors:9*
T0*
_output_shapes
:2

Identity_9Щ
AssignVariableOp_9AssignVariableOp#assignvariableop_9_b_norm_1_14_betaIdentity_9:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_9_
Identity_10IdentityRestoreV2:tensors:10*
T0*
_output_shapes
:2
Identity_10д
AssignVariableOp_10AssignVariableOp+assignvariableop_10_b_norm_1_14_moving_meanIdentity_10:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_10_
Identity_11IdentityRestoreV2:tensors:11*
T0*
_output_shapes
:2
Identity_11и
AssignVariableOp_11AssignVariableOp/assignvariableop_11_b_norm_1_14_moving_varianceIdentity_11:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_11_
Identity_12IdentityRestoreV2:tensors:12*
T0*
_output_shapes
:2
Identity_12Э
AssignVariableOp_12AssignVariableOp$assignvariableop_12_conv_2_14_kernelIdentity_12:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_12_
Identity_13IdentityRestoreV2:tensors:13*
T0*
_output_shapes
:2
Identity_13Ы
AssignVariableOp_13AssignVariableOp"assignvariableop_13_conv_2_14_biasIdentity_13:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_13_
Identity_14IdentityRestoreV2:tensors:14*
T0*
_output_shapes
:2
Identity_14Ю
AssignVariableOp_14AssignVariableOp%assignvariableop_14_b_norm_2_14_gammaIdentity_14:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_14_
Identity_15IdentityRestoreV2:tensors:15*
T0*
_output_shapes
:2
Identity_15Э
AssignVariableOp_15AssignVariableOp$assignvariableop_15_b_norm_2_14_betaIdentity_15:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_15_
Identity_16IdentityRestoreV2:tensors:16*
T0*
_output_shapes
:2
Identity_16д
AssignVariableOp_16AssignVariableOp+assignvariableop_16_b_norm_2_14_moving_meanIdentity_16:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_16_
Identity_17IdentityRestoreV2:tensors:17*
T0*
_output_shapes
:2
Identity_17и
AssignVariableOp_17AssignVariableOp/assignvariableop_17_b_norm_2_14_moving_varianceIdentity_17:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_17_
Identity_18IdentityRestoreV2:tensors:18*
T0*
_output_shapes
:2
Identity_18Ь
AssignVariableOp_18AssignVariableOp#assignvariableop_18_conv_3_9_kernelIdentity_18:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_18_
Identity_19IdentityRestoreV2:tensors:19*
T0*
_output_shapes
:2
Identity_19Ъ
AssignVariableOp_19AssignVariableOp!assignvariableop_19_conv_3_9_biasIdentity_19:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_19_
Identity_20IdentityRestoreV2:tensors:20*
T0*
_output_shapes
:2
Identity_20Э
AssignVariableOp_20AssignVariableOp$assignvariableop_20_b_norm_3_9_gammaIdentity_20:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_20_
Identity_21IdentityRestoreV2:tensors:21*
T0*
_output_shapes
:2
Identity_21Ь
AssignVariableOp_21AssignVariableOp#assignvariableop_21_b_norm_3_9_betaIdentity_21:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_21_
Identity_22IdentityRestoreV2:tensors:22*
T0*
_output_shapes
:2
Identity_22г
AssignVariableOp_22AssignVariableOp*assignvariableop_22_b_norm_3_9_moving_meanIdentity_22:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_22_
Identity_23IdentityRestoreV2:tensors:23*
T0*
_output_shapes
:2
Identity_23з
AssignVariableOp_23AssignVariableOp.assignvariableop_23_b_norm_3_9_moving_varianceIdentity_23:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_23_
Identity_24IdentityRestoreV2:tensors:24*
T0*
_output_shapes
:2
Identity_24Ь
AssignVariableOp_24AssignVariableOp#assignvariableop_24_conv_4_4_kernelIdentity_24:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_24_
Identity_25IdentityRestoreV2:tensors:25*
T0*
_output_shapes
:2
Identity_25Ъ
AssignVariableOp_25AssignVariableOp!assignvariableop_25_conv_4_4_biasIdentity_25:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_25_
Identity_26IdentityRestoreV2:tensors:26*
T0*
_output_shapes
:2
Identity_26Э
AssignVariableOp_26AssignVariableOp$assignvariableop_26_b_norm_4_4_gammaIdentity_26:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_26_
Identity_27IdentityRestoreV2:tensors:27*
T0*
_output_shapes
:2
Identity_27Ь
AssignVariableOp_27AssignVariableOp#assignvariableop_27_b_norm_4_4_betaIdentity_27:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_27_
Identity_28IdentityRestoreV2:tensors:28*
T0*
_output_shapes
:2
Identity_28г
AssignVariableOp_28AssignVariableOp*assignvariableop_28_b_norm_4_4_moving_meanIdentity_28:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_28_
Identity_29IdentityRestoreV2:tensors:29*
T0*
_output_shapes
:2
Identity_29з
AssignVariableOp_29AssignVariableOp.assignvariableop_29_b_norm_4_4_moving_varianceIdentity_29:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_29_
Identity_30IdentityRestoreV2:tensors:30*
T0*
_output_shapes
:2
Identity_30░
AssignVariableOp_30AssignVariableOp7assignvariableop_30_output_class_distribution_14_kernelIdentity_30:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_30_
Identity_31IdentityRestoreV2:tensors:31*
T0*
_output_shapes
:2
Identity_31о
AssignVariableOp_31AssignVariableOp5assignvariableop_31_output_class_distribution_14_biasIdentity_31:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_31_
Identity_32IdentityRestoreV2:tensors:32*
T0	*
_output_shapes
:2
Identity_32Ц
AssignVariableOp_32AssignVariableOpassignvariableop_32_adam_iterIdentity_32:output:0*
_output_shapes
 *
dtype0	2
AssignVariableOp_32_
Identity_33IdentityRestoreV2:tensors:33*
T0*
_output_shapes
:2
Identity_33Ш
AssignVariableOp_33AssignVariableOpassignvariableop_33_adam_beta_1Identity_33:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_33_
Identity_34IdentityRestoreV2:tensors:34*
T0*
_output_shapes
:2
Identity_34Ш
AssignVariableOp_34AssignVariableOpassignvariableop_34_adam_beta_2Identity_34:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_34_
Identity_35IdentityRestoreV2:tensors:35*
T0*
_output_shapes
:2
Identity_35Ч
AssignVariableOp_35AssignVariableOpassignvariableop_35_adam_decayIdentity_35:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_35_
Identity_36IdentityRestoreV2:tensors:36*
T0*
_output_shapes
:2
Identity_36Я
AssignVariableOp_36AssignVariableOp&assignvariableop_36_adam_learning_rateIdentity_36:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_36_
Identity_37IdentityRestoreV2:tensors:37*
T0*
_output_shapes
:2
Identity_37Т
AssignVariableOp_37AssignVariableOpassignvariableop_37_totalIdentity_37:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_37_
Identity_38IdentityRestoreV2:tensors:38*
T0*
_output_shapes
:2
Identity_38Т
AssignVariableOp_38AssignVariableOpassignvariableop_38_countIdentity_38:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_38_
Identity_39IdentityRestoreV2:tensors:39*
T0*
_output_shapes
:2
Identity_39и
AssignVariableOp_39AssignVariableOp/assignvariableop_39_adam_input_conv_14_kernel_mIdentity_39:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_39_
Identity_40IdentityRestoreV2:tensors:40*
T0*
_output_shapes
:2
Identity_40ж
AssignVariableOp_40AssignVariableOp-assignvariableop_40_adam_input_conv_14_bias_mIdentity_40:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_40_
Identity_41IdentityRestoreV2:tensors:41*
T0*
_output_shapes
:2
Identity_41╢
AssignVariableOp_41AssignVariableOp=assignvariableop_41_adam_input_batch_normalization_14_gamma_mIdentity_41:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_41_
Identity_42IdentityRestoreV2:tensors:42*
T0*
_output_shapes
:2
Identity_42╡
AssignVariableOp_42AssignVariableOp<assignvariableop_42_adam_input_batch_normalization_14_beta_mIdentity_42:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_42_
Identity_43IdentityRestoreV2:tensors:43*
T0*
_output_shapes
:2
Identity_43д
AssignVariableOp_43AssignVariableOp+assignvariableop_43_adam_conv_1_14_kernel_mIdentity_43:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_43_
Identity_44IdentityRestoreV2:tensors:44*
T0*
_output_shapes
:2
Identity_44в
AssignVariableOp_44AssignVariableOp)assignvariableop_44_adam_conv_1_14_bias_mIdentity_44:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_44_
Identity_45IdentityRestoreV2:tensors:45*
T0*
_output_shapes
:2
Identity_45е
AssignVariableOp_45AssignVariableOp,assignvariableop_45_adam_b_norm_1_14_gamma_mIdentity_45:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_45_
Identity_46IdentityRestoreV2:tensors:46*
T0*
_output_shapes
:2
Identity_46д
AssignVariableOp_46AssignVariableOp+assignvariableop_46_adam_b_norm_1_14_beta_mIdentity_46:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_46_
Identity_47IdentityRestoreV2:tensors:47*
T0*
_output_shapes
:2
Identity_47д
AssignVariableOp_47AssignVariableOp+assignvariableop_47_adam_conv_2_14_kernel_mIdentity_47:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_47_
Identity_48IdentityRestoreV2:tensors:48*
T0*
_output_shapes
:2
Identity_48в
AssignVariableOp_48AssignVariableOp)assignvariableop_48_adam_conv_2_14_bias_mIdentity_48:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_48_
Identity_49IdentityRestoreV2:tensors:49*
T0*
_output_shapes
:2
Identity_49е
AssignVariableOp_49AssignVariableOp,assignvariableop_49_adam_b_norm_2_14_gamma_mIdentity_49:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_49_
Identity_50IdentityRestoreV2:tensors:50*
T0*
_output_shapes
:2
Identity_50д
AssignVariableOp_50AssignVariableOp+assignvariableop_50_adam_b_norm_2_14_beta_mIdentity_50:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_50_
Identity_51IdentityRestoreV2:tensors:51*
T0*
_output_shapes
:2
Identity_51г
AssignVariableOp_51AssignVariableOp*assignvariableop_51_adam_conv_3_9_kernel_mIdentity_51:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_51_
Identity_52IdentityRestoreV2:tensors:52*
T0*
_output_shapes
:2
Identity_52б
AssignVariableOp_52AssignVariableOp(assignvariableop_52_adam_conv_3_9_bias_mIdentity_52:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_52_
Identity_53IdentityRestoreV2:tensors:53*
T0*
_output_shapes
:2
Identity_53д
AssignVariableOp_53AssignVariableOp+assignvariableop_53_adam_b_norm_3_9_gamma_mIdentity_53:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_53_
Identity_54IdentityRestoreV2:tensors:54*
T0*
_output_shapes
:2
Identity_54г
AssignVariableOp_54AssignVariableOp*assignvariableop_54_adam_b_norm_3_9_beta_mIdentity_54:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_54_
Identity_55IdentityRestoreV2:tensors:55*
T0*
_output_shapes
:2
Identity_55г
AssignVariableOp_55AssignVariableOp*assignvariableop_55_adam_conv_4_4_kernel_mIdentity_55:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_55_
Identity_56IdentityRestoreV2:tensors:56*
T0*
_output_shapes
:2
Identity_56б
AssignVariableOp_56AssignVariableOp(assignvariableop_56_adam_conv_4_4_bias_mIdentity_56:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_56_
Identity_57IdentityRestoreV2:tensors:57*
T0*
_output_shapes
:2
Identity_57д
AssignVariableOp_57AssignVariableOp+assignvariableop_57_adam_b_norm_4_4_gamma_mIdentity_57:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_57_
Identity_58IdentityRestoreV2:tensors:58*
T0*
_output_shapes
:2
Identity_58г
AssignVariableOp_58AssignVariableOp*assignvariableop_58_adam_b_norm_4_4_beta_mIdentity_58:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_58_
Identity_59IdentityRestoreV2:tensors:59*
T0*
_output_shapes
:2
Identity_59╖
AssignVariableOp_59AssignVariableOp>assignvariableop_59_adam_output_class_distribution_14_kernel_mIdentity_59:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_59_
Identity_60IdentityRestoreV2:tensors:60*
T0*
_output_shapes
:2
Identity_60╡
AssignVariableOp_60AssignVariableOp<assignvariableop_60_adam_output_class_distribution_14_bias_mIdentity_60:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_60_
Identity_61IdentityRestoreV2:tensors:61*
T0*
_output_shapes
:2
Identity_61и
AssignVariableOp_61AssignVariableOp/assignvariableop_61_adam_input_conv_14_kernel_vIdentity_61:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_61_
Identity_62IdentityRestoreV2:tensors:62*
T0*
_output_shapes
:2
Identity_62ж
AssignVariableOp_62AssignVariableOp-assignvariableop_62_adam_input_conv_14_bias_vIdentity_62:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_62_
Identity_63IdentityRestoreV2:tensors:63*
T0*
_output_shapes
:2
Identity_63╢
AssignVariableOp_63AssignVariableOp=assignvariableop_63_adam_input_batch_normalization_14_gamma_vIdentity_63:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_63_
Identity_64IdentityRestoreV2:tensors:64*
T0*
_output_shapes
:2
Identity_64╡
AssignVariableOp_64AssignVariableOp<assignvariableop_64_adam_input_batch_normalization_14_beta_vIdentity_64:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_64_
Identity_65IdentityRestoreV2:tensors:65*
T0*
_output_shapes
:2
Identity_65д
AssignVariableOp_65AssignVariableOp+assignvariableop_65_adam_conv_1_14_kernel_vIdentity_65:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_65_
Identity_66IdentityRestoreV2:tensors:66*
T0*
_output_shapes
:2
Identity_66в
AssignVariableOp_66AssignVariableOp)assignvariableop_66_adam_conv_1_14_bias_vIdentity_66:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_66_
Identity_67IdentityRestoreV2:tensors:67*
T0*
_output_shapes
:2
Identity_67е
AssignVariableOp_67AssignVariableOp,assignvariableop_67_adam_b_norm_1_14_gamma_vIdentity_67:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_67_
Identity_68IdentityRestoreV2:tensors:68*
T0*
_output_shapes
:2
Identity_68д
AssignVariableOp_68AssignVariableOp+assignvariableop_68_adam_b_norm_1_14_beta_vIdentity_68:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_68_
Identity_69IdentityRestoreV2:tensors:69*
T0*
_output_shapes
:2
Identity_69д
AssignVariableOp_69AssignVariableOp+assignvariableop_69_adam_conv_2_14_kernel_vIdentity_69:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_69_
Identity_70IdentityRestoreV2:tensors:70*
T0*
_output_shapes
:2
Identity_70в
AssignVariableOp_70AssignVariableOp)assignvariableop_70_adam_conv_2_14_bias_vIdentity_70:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_70_
Identity_71IdentityRestoreV2:tensors:71*
T0*
_output_shapes
:2
Identity_71е
AssignVariableOp_71AssignVariableOp,assignvariableop_71_adam_b_norm_2_14_gamma_vIdentity_71:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_71_
Identity_72IdentityRestoreV2:tensors:72*
T0*
_output_shapes
:2
Identity_72д
AssignVariableOp_72AssignVariableOp+assignvariableop_72_adam_b_norm_2_14_beta_vIdentity_72:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_72_
Identity_73IdentityRestoreV2:tensors:73*
T0*
_output_shapes
:2
Identity_73г
AssignVariableOp_73AssignVariableOp*assignvariableop_73_adam_conv_3_9_kernel_vIdentity_73:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_73_
Identity_74IdentityRestoreV2:tensors:74*
T0*
_output_shapes
:2
Identity_74б
AssignVariableOp_74AssignVariableOp(assignvariableop_74_adam_conv_3_9_bias_vIdentity_74:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_74_
Identity_75IdentityRestoreV2:tensors:75*
T0*
_output_shapes
:2
Identity_75д
AssignVariableOp_75AssignVariableOp+assignvariableop_75_adam_b_norm_3_9_gamma_vIdentity_75:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_75_
Identity_76IdentityRestoreV2:tensors:76*
T0*
_output_shapes
:2
Identity_76г
AssignVariableOp_76AssignVariableOp*assignvariableop_76_adam_b_norm_3_9_beta_vIdentity_76:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_76_
Identity_77IdentityRestoreV2:tensors:77*
T0*
_output_shapes
:2
Identity_77г
AssignVariableOp_77AssignVariableOp*assignvariableop_77_adam_conv_4_4_kernel_vIdentity_77:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_77_
Identity_78IdentityRestoreV2:tensors:78*
T0*
_output_shapes
:2
Identity_78б
AssignVariableOp_78AssignVariableOp(assignvariableop_78_adam_conv_4_4_bias_vIdentity_78:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_78_
Identity_79IdentityRestoreV2:tensors:79*
T0*
_output_shapes
:2
Identity_79д
AssignVariableOp_79AssignVariableOp+assignvariableop_79_adam_b_norm_4_4_gamma_vIdentity_79:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_79_
Identity_80IdentityRestoreV2:tensors:80*
T0*
_output_shapes
:2
Identity_80г
AssignVariableOp_80AssignVariableOp*assignvariableop_80_adam_b_norm_4_4_beta_vIdentity_80:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_80_
Identity_81IdentityRestoreV2:tensors:81*
T0*
_output_shapes
:2
Identity_81╖
AssignVariableOp_81AssignVariableOp>assignvariableop_81_adam_output_class_distribution_14_kernel_vIdentity_81:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_81_
Identity_82IdentityRestoreV2:tensors:82*
T0*
_output_shapes
:2
Identity_82╡
AssignVariableOp_82AssignVariableOp<assignvariableop_82_adam_output_class_distribution_14_bias_vIdentity_82:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_82и
RestoreV2_1/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*1
value(B&B_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2_1/tensor_namesФ
RestoreV2_1/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*
valueB
B 2
RestoreV2_1/shape_and_slices─
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
NoOpА
Identity_83Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_55^AssignVariableOp_56^AssignVariableOp_57^AssignVariableOp_58^AssignVariableOp_59^AssignVariableOp_6^AssignVariableOp_60^AssignVariableOp_61^AssignVariableOp_62^AssignVariableOp_63^AssignVariableOp_64^AssignVariableOp_65^AssignVariableOp_66^AssignVariableOp_67^AssignVariableOp_68^AssignVariableOp_69^AssignVariableOp_7^AssignVariableOp_70^AssignVariableOp_71^AssignVariableOp_72^AssignVariableOp_73^AssignVariableOp_74^AssignVariableOp_75^AssignVariableOp_76^AssignVariableOp_77^AssignVariableOp_78^AssignVariableOp_79^AssignVariableOp_8^AssignVariableOp_80^AssignVariableOp_81^AssignVariableOp_82^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2
Identity_83Н
Identity_84IdentityIdentity_83:output:0^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_55^AssignVariableOp_56^AssignVariableOp_57^AssignVariableOp_58^AssignVariableOp_59^AssignVariableOp_6^AssignVariableOp_60^AssignVariableOp_61^AssignVariableOp_62^AssignVariableOp_63^AssignVariableOp_64^AssignVariableOp_65^AssignVariableOp_66^AssignVariableOp_67^AssignVariableOp_68^AssignVariableOp_69^AssignVariableOp_7^AssignVariableOp_70^AssignVariableOp_71^AssignVariableOp_72^AssignVariableOp_73^AssignVariableOp_74^AssignVariableOp_75^AssignVariableOp_76^AssignVariableOp_77^AssignVariableOp_78^AssignVariableOp_79^AssignVariableOp_8^AssignVariableOp_80^AssignVariableOp_81^AssignVariableOp_82^AssignVariableOp_9
^RestoreV2^RestoreV2_1*
T0*
_output_shapes
: 2
Identity_84"#
identity_84Identity_84:output:0*у
_input_shapes╤
╬: :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::2$
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
∙
Д
;__inference_input_batch_normalization_layer_call_fn_2737688

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4
identityИвStatefulPartitionedCallў
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4*
Tin	
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*A
_output_shapes/
-:+                            *-
config_proto

GPU

CPU2*0J 8*_
fZRX
V__inference_input_batch_normalization_layer_call_and_return_conditional_losses_27354432
StatefulPartitionedCallи
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+                            2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+                            ::::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs
ї
ш
E__inference_b_norm_1_layer_call_and_return_conditional_losses_2735607

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityИвFusedBatchNormV3/ReadVariableOpв!FusedBatchNormV3/ReadVariableOp_1вReadVariableOpвReadVariableOp_1^
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
ReadVariableOp_1з
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02!
FusedBatchNormV3/ReadVariableOpн
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02#
!FusedBatchNormV3/ReadVariableOp_1▄
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+                            : : : : :*
epsilon%oГ:*
is_training( 2
FusedBatchNormV3S
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *дp}?2
Constь
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*A
_output_shapes/
-:+                            2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+                            ::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:& "
 
_user_specified_nameinputs
Ж
∙
V__inference_input_batch_normalization_layer_call_and_return_conditional_losses_2737670

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityИвFusedBatchNormV3/ReadVariableOpв!FusedBatchNormV3/ReadVariableOp_1вReadVariableOpвReadVariableOp_1^
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
ReadVariableOp_1з
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02!
FusedBatchNormV3/ReadVariableOpн
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02#
!FusedBatchNormV3/ReadVariableOp_1▄
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+                            : : : : :*
epsilon%oГ:*
is_training( 2
FusedBatchNormV3S
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *дp}?2
Constь
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*A
_output_shapes/
-:+                            2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+                            ::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:& "
 
_user_specified_nameinputs
┴
й
(__inference_conv_2_layer_call_fn_2735646

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2
identityИвStatefulPartitionedCallв
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*A
_output_shapes/
-:+                           @*-
config_proto

GPU

CPU2*0J 8*L
fGRE
C__inference_conv_2_layer_call_and_return_conditional_losses_27356382
StatefulPartitionedCallи
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+                           @2

Identity"
identityIdentity:output:0*H
_input_shapes7
5:+                            ::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs
╫
є
*__inference_b_norm_2_layer_call_fn_2737989

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4
identityИвStatefulPartitionedCallц
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4*
Tin	
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*A
_output_shapes/
-:+                           @*-
config_proto

GPU

CPU2*0J 8*N
fIRG
E__inference_b_norm_2_layer_call_and_return_conditional_losses_27357712
StatefulPartitionedCallи
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+                           @2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+                           @::::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs"пL
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*┘
serving_default┼
W
input_conv_inputC
"serving_default_input_conv_input:0         ААN
output_class_distribution1
StatefulPartitionedCall:0         ╚tensorflow/serving/predict:╘┬
ЯП
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
	variables
regularization_losses
trainable_variables
	keras_api

signatures
+╞&call_and_return_all_conditional_losses
╟__call__
╚_default_save_signature"▌И
_tf_keras_sequential╜И{"class_name": "Sequential", "name": "awsome_net", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "config": {"name": "awsome_net", "layers": [{"class_name": "Conv2D", "config": {"name": "input_conv", "trainable": true, "batch_input_shape": [null, 128, 128, 3], "dtype": "float32", "filters": 32, "kernel_size": [3, 3], "strides": [1, 1], "padding": "same", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "BatchNormalization", "config": {"name": "input_batch_normalization", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}}, {"class_name": "ReLU", "config": {"name": "input_RELU", "trainable": true, "dtype": "float32", "max_value": null, "negative_slope": 0.0, "threshold": 0.0}}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_60", "trainable": true, "dtype": "float32", "pool_size": [2, 2], "padding": "valid", "strides": [2, 2], "data_format": "channels_last"}}, {"class_name": "Dropout", "config": {"name": "dropout_28", "trainable": true, "dtype": "float32", "rate": 0.25, "noise_shape": null, "seed": null}}, {"class_name": "Conv2D", "config": {"name": "conv_1", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": [3, 3], "strides": [1, 1], "padding": "same", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "BatchNormalization", "config": {"name": "b_norm_1", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}}, {"class_name": "ReLU", "config": {"name": "RELU_1", "trainable": true, "dtype": "float32", "max_value": null, "negative_slope": 0.0, "threshold": 0.0}}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_56", "trainable": true, "dtype": "float32", "pool_size": [2, 2], "padding": "valid", "strides": [2, 2], "data_format": "channels_last"}}, {"class_name": "Conv2D", "config": {"name": "conv_2", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": [3, 3], "strides": [1, 1], "padding": "same", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "BatchNormalization", "config": {"name": "b_norm_2", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}}, {"class_name": "ReLU", "config": {"name": "RELU_2", "trainable": true, "dtype": "float32", "max_value": null, "negative_slope": 0.0, "threshold": 0.0}}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_57", "trainable": true, "dtype": "float32", "pool_size": [2, 2], "padding": "valid", "strides": [2, 2], "data_format": "channels_last"}}, {"class_name": "Conv2D", "config": {"name": "conv_3", "trainable": true, "dtype": "float32", "filters": 128, "kernel_size": [3, 3], "strides": [1, 1], "padding": "same", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "BatchNormalization", "config": {"name": "b_norm_3", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}}, {"class_name": "ReLU", "config": {"name": "RELU_3", "trainable": true, "dtype": "float32", "max_value": null, "negative_slope": 0.0, "threshold": 0.0}}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_58", "trainable": true, "dtype": "float32", "pool_size": [2, 2], "padding": "valid", "strides": [2, 2], "data_format": "channels_last"}}, {"class_name": "Conv2D", "config": {"name": "conv_4", "trainable": true, "dtype": "float32", "filters": 128, "kernel_size": [3, 3], "strides": [1, 1], "padding": "same", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "BatchNormalization", "config": {"name": "b_norm_4", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}}, {"class_name": "ReLU", "config": {"name": "RELU_4", "trainable": true, "dtype": "float32", "max_value": null, "negative_slope": 0.0, "threshold": 0.0}}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_59", "trainable": true, "dtype": "float32", "pool_size": [2, 2], "padding": "valid", "strides": [2, 2], "data_format": "channels_last"}}, {"class_name": "Flatten", "config": {"name": "flatten_14", "trainable": true, "dtype": "float32", "data_format": "channels_last"}}, {"class_name": "Dropout", "config": {"name": "dropout_29", "trainable": true, "dtype": "float32", "rate": 0.25, "noise_shape": null, "seed": null}}, {"class_name": "Dense", "config": {"name": "output_class_distribution", "trainable": true, "dtype": "float32", "units": 200, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}]}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"-1": 3}}}, "is_graph_network": true, "keras_version": "2.2.4-tf", "backend": "tensorflow", "model_config": {"class_name": "Sequential", "config": {"name": "awsome_net", "layers": [{"class_name": "Conv2D", "config": {"name": "input_conv", "trainable": true, "batch_input_shape": [null, 128, 128, 3], "dtype": "float32", "filters": 32, "kernel_size": [3, 3], "strides": [1, 1], "padding": "same", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "BatchNormalization", "config": {"name": "input_batch_normalization", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}}, {"class_name": "ReLU", "config": {"name": "input_RELU", "trainable": true, "dtype": "float32", "max_value": null, "negative_slope": 0.0, "threshold": 0.0}}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_60", "trainable": true, "dtype": "float32", "pool_size": [2, 2], "padding": "valid", "strides": [2, 2], "data_format": "channels_last"}}, {"class_name": "Dropout", "config": {"name": "dropout_28", "trainable": true, "dtype": "float32", "rate": 0.25, "noise_shape": null, "seed": null}}, {"class_name": "Conv2D", "config": {"name": "conv_1", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": [3, 3], "strides": [1, 1], "padding": "same", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "BatchNormalization", "config": {"name": "b_norm_1", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}}, {"class_name": "ReLU", "config": {"name": "RELU_1", "trainable": true, "dtype": "float32", "max_value": null, "negative_slope": 0.0, "threshold": 0.0}}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_56", "trainable": true, "dtype": "float32", "pool_size": [2, 2], "padding": "valid", "strides": [2, 2], "data_format": "channels_last"}}, {"class_name": "Conv2D", "config": {"name": "conv_2", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": [3, 3], "strides": [1, 1], "padding": "same", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "BatchNormalization", "config": {"name": "b_norm_2", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}}, {"class_name": "ReLU", "config": {"name": "RELU_2", "trainable": true, "dtype": "float32", "max_value": null, "negative_slope": 0.0, "threshold": 0.0}}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_57", "trainable": true, "dtype": "float32", "pool_size": [2, 2], "padding": "valid", "strides": [2, 2], "data_format": "channels_last"}}, {"class_name": "Conv2D", "config": {"name": "conv_3", "trainable": true, "dtype": "float32", "filters": 128, "kernel_size": [3, 3], "strides": [1, 1], "padding": "same", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "BatchNormalization", "config": {"name": "b_norm_3", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}}, {"class_name": "ReLU", "config": {"name": "RELU_3", "trainable": true, "dtype": "float32", "max_value": null, "negative_slope": 0.0, "threshold": 0.0}}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_58", "trainable": true, "dtype": "float32", "pool_size": [2, 2], "padding": "valid", "strides": [2, 2], "data_format": "channels_last"}}, {"class_name": "Conv2D", "config": {"name": "conv_4", "trainable": true, "dtype": "float32", "filters": 128, "kernel_size": [3, 3], "strides": [1, 1], "padding": "same", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "BatchNormalization", "config": {"name": "b_norm_4", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}}, {"class_name": "ReLU", "config": {"name": "RELU_4", "trainable": true, "dtype": "float32", "max_value": null, "negative_slope": 0.0, "threshold": 0.0}}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_59", "trainable": true, "dtype": "float32", "pool_size": [2, 2], "padding": "valid", "strides": [2, 2], "data_format": "channels_last"}}, {"class_name": "Flatten", "config": {"name": "flatten_14", "trainable": true, "dtype": "float32", "data_format": "channels_last"}}, {"class_name": "Dropout", "config": {"name": "dropout_29", "trainable": true, "dtype": "float32", "rate": 0.25, "noise_shape": null, "seed": null}}, {"class_name": "Dense", "config": {"name": "output_class_distribution", "trainable": true, "dtype": "float32", "units": 200, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}]}}, "training_config": {"loss": {"class_name": "CategoricalCrossentropy", "config": {"reduction": "auto", "name": "categorical_crossentropy", "from_logits": true, "label_smoothing": 0}}, "metrics": [{"class_name": "CategoricalAccuracy", "config": {"name": "categorical_accuracy", "dtype": "float32"}}], "weighted_metrics": null, "sample_weight_mode": null, "loss_weights": null, "optimizer_config": {"class_name": "Adam", "config": {"name": "Adam", "learning_rate": 0.0002575, "decay": 0.0, "beta_1": 0.8999999761581421, "beta_2": 0.9990000128746033, "epsilon": 1e-07, "amsgrad": false}}}}
├"└
_tf_keras_input_layerа{"class_name": "InputLayer", "name": "input_conv_input", "dtype": "float32", "sparse": false, "ragged": false, "batch_input_shape": [null, 128, 128, 3], "config": {"batch_input_shape": [null, 128, 128, 3], "dtype": "float32", "sparse": false, "ragged": false, "name": "input_conv_input"}}
м

 kernel
!bias
"	variables
#regularization_losses
$trainable_variables
%	keras_api
+╔&call_and_return_all_conditional_losses
╩__call__"Е
_tf_keras_layerы{"class_name": "Conv2D", "name": "input_conv", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": [null, 128, 128, 3], "config": {"name": "input_conv", "trainable": true, "batch_input_shape": [null, 128, 128, 3], "dtype": "float32", "filters": 32, "kernel_size": [3, 3], "strides": [1, 1], "padding": "same", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"-1": 3}}}}
╜
&axis
	'gamma
(beta
)moving_mean
*moving_variance
+	variables
,regularization_losses
-trainable_variables
.	keras_api
+╦&call_and_return_all_conditional_losses
╠__call__"ч
_tf_keras_layer═{"class_name": "BatchNormalization", "name": "input_batch_normalization", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "config": {"name": "input_batch_normalization", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"3": 32}}}}
╜
/	variables
0regularization_losses
1trainable_variables
2	keras_api
+═&call_and_return_all_conditional_losses
╬__call__"м
_tf_keras_layerТ{"class_name": "ReLU", "name": "input_RELU", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "input_RELU", "trainable": true, "dtype": "float32", "max_value": null, "negative_slope": 0.0, "threshold": 0.0}}
Б
3	variables
4regularization_losses
5trainable_variables
6	keras_api
+╧&call_and_return_all_conditional_losses
╨__call__"Ё
_tf_keras_layer╓{"class_name": "MaxPooling2D", "name": "max_pooling2d_60", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "max_pooling2d_60", "trainable": true, "dtype": "float32", "pool_size": [2, 2], "padding": "valid", "strides": [2, 2], "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}}
┤
7	variables
8regularization_losses
9trainable_variables
:	keras_api
+╤&call_and_return_all_conditional_losses
╥__call__"г
_tf_keras_layerЙ{"class_name": "Dropout", "name": "dropout_28", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "config": {"name": "dropout_28", "trainable": true, "dtype": "float32", "rate": 0.25, "noise_shape": null, "seed": null}}
ь

;kernel
<bias
=	variables
>regularization_losses
?trainable_variables
@	keras_api
+╙&call_and_return_all_conditional_losses
╘__call__"┼
_tf_keras_layerл{"class_name": "Conv2D", "name": "conv_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "conv_1", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": [3, 3], "strides": [1, 1], "padding": "same", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"-1": 32}}}}
Ы
Aaxis
	Bgamma
Cbeta
Dmoving_mean
Emoving_variance
F	variables
Gregularization_losses
Htrainable_variables
I	keras_api
+╒&call_and_return_all_conditional_losses
╓__call__"┼
_tf_keras_layerл{"class_name": "BatchNormalization", "name": "b_norm_1", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "config": {"name": "b_norm_1", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"3": 32}}}}
╡
J	variables
Kregularization_losses
Ltrainable_variables
M	keras_api
+╫&call_and_return_all_conditional_losses
╪__call__"д
_tf_keras_layerК{"class_name": "ReLU", "name": "RELU_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "RELU_1", "trainable": true, "dtype": "float32", "max_value": null, "negative_slope": 0.0, "threshold": 0.0}}
Б
N	variables
Oregularization_losses
Ptrainable_variables
Q	keras_api
+┘&call_and_return_all_conditional_losses
┌__call__"Ё
_tf_keras_layer╓{"class_name": "MaxPooling2D", "name": "max_pooling2d_56", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "max_pooling2d_56", "trainable": true, "dtype": "float32", "pool_size": [2, 2], "padding": "valid", "strides": [2, 2], "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}}
ь

Rkernel
Sbias
T	variables
Uregularization_losses
Vtrainable_variables
W	keras_api
+█&call_and_return_all_conditional_losses
▄__call__"┼
_tf_keras_layerл{"class_name": "Conv2D", "name": "conv_2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "conv_2", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": [3, 3], "strides": [1, 1], "padding": "same", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"-1": 32}}}}
Ы
Xaxis
	Ygamma
Zbeta
[moving_mean
\moving_variance
]	variables
^regularization_losses
_trainable_variables
`	keras_api
+▌&call_and_return_all_conditional_losses
▐__call__"┼
_tf_keras_layerл{"class_name": "BatchNormalization", "name": "b_norm_2", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "config": {"name": "b_norm_2", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"3": 64}}}}
╡
a	variables
bregularization_losses
ctrainable_variables
d	keras_api
+▀&call_and_return_all_conditional_losses
р__call__"д
_tf_keras_layerК{"class_name": "ReLU", "name": "RELU_2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "RELU_2", "trainable": true, "dtype": "float32", "max_value": null, "negative_slope": 0.0, "threshold": 0.0}}
Б
e	variables
fregularization_losses
gtrainable_variables
h	keras_api
+с&call_and_return_all_conditional_losses
т__call__"Ё
_tf_keras_layer╓{"class_name": "MaxPooling2D", "name": "max_pooling2d_57", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "max_pooling2d_57", "trainable": true, "dtype": "float32", "pool_size": [2, 2], "padding": "valid", "strides": [2, 2], "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}}
э

ikernel
jbias
k	variables
lregularization_losses
mtrainable_variables
n	keras_api
+у&call_and_return_all_conditional_losses
ф__call__"╞
_tf_keras_layerм{"class_name": "Conv2D", "name": "conv_3", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "conv_3", "trainable": true, "dtype": "float32", "filters": 128, "kernel_size": [3, 3], "strides": [1, 1], "padding": "same", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"-1": 64}}}}
Ь
oaxis
	pgamma
qbeta
rmoving_mean
smoving_variance
t	variables
uregularization_losses
vtrainable_variables
w	keras_api
+х&call_and_return_all_conditional_losses
ц__call__"╞
_tf_keras_layerм{"class_name": "BatchNormalization", "name": "b_norm_3", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "config": {"name": "b_norm_3", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"3": 128}}}}
╡
x	variables
yregularization_losses
ztrainable_variables
{	keras_api
+ч&call_and_return_all_conditional_losses
ш__call__"д
_tf_keras_layerК{"class_name": "ReLU", "name": "RELU_3", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "RELU_3", "trainable": true, "dtype": "float32", "max_value": null, "negative_slope": 0.0, "threshold": 0.0}}
Б
|	variables
}regularization_losses
~trainable_variables
	keras_api
+щ&call_and_return_all_conditional_losses
ъ__call__"Ё
_tf_keras_layer╓{"class_name": "MaxPooling2D", "name": "max_pooling2d_58", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "max_pooling2d_58", "trainable": true, "dtype": "float32", "pool_size": [2, 2], "padding": "valid", "strides": [2, 2], "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}}
Ї
Аkernel
	Бbias
В	variables
Гregularization_losses
Дtrainable_variables
Е	keras_api
+ы&call_and_return_all_conditional_losses
ь__call__"╟
_tf_keras_layerн{"class_name": "Conv2D", "name": "conv_4", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "conv_4", "trainable": true, "dtype": "float32", "filters": 128, "kernel_size": [3, 3], "strides": [1, 1], "padding": "same", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"-1": 128}}}}
е
	Жaxis

Зgamma
	Иbeta
Йmoving_mean
Кmoving_variance
Л	variables
Мregularization_losses
Нtrainable_variables
О	keras_api
+э&call_and_return_all_conditional_losses
ю__call__"╞
_tf_keras_layerм{"class_name": "BatchNormalization", "name": "b_norm_4", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "config": {"name": "b_norm_4", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"3": 128}}}}
╣
П	variables
Рregularization_losses
Сtrainable_variables
Т	keras_api
+я&call_and_return_all_conditional_losses
Ё__call__"д
_tf_keras_layerК{"class_name": "ReLU", "name": "RELU_4", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "RELU_4", "trainable": true, "dtype": "float32", "max_value": null, "negative_slope": 0.0, "threshold": 0.0}}
Е
У	variables
Фregularization_losses
Хtrainable_variables
Ц	keras_api
+ё&call_and_return_all_conditional_losses
Є__call__"Ё
_tf_keras_layer╓{"class_name": "MaxPooling2D", "name": "max_pooling2d_59", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "max_pooling2d_59", "trainable": true, "dtype": "float32", "pool_size": [2, 2], "padding": "valid", "strides": [2, 2], "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}}
╕
Ч	variables
Шregularization_losses
Щtrainable_variables
Ъ	keras_api
+є&call_and_return_all_conditional_losses
Ї__call__"г
_tf_keras_layerЙ{"class_name": "Flatten", "name": "flatten_14", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "flatten_14", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 1, "axes": {}}}}
╕
Ы	variables
Ьregularization_losses
Эtrainable_variables
Ю	keras_api
+ї&call_and_return_all_conditional_losses
Ў__call__"г
_tf_keras_layerЙ{"class_name": "Dropout", "name": "dropout_29", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "config": {"name": "dropout_29", "trainable": true, "dtype": "float32", "rate": 0.25, "noise_shape": null, "seed": null}}
в
Яkernel
	аbias
б	variables
вregularization_losses
гtrainable_variables
д	keras_api
+ў&call_and_return_all_conditional_losses
°__call__"ї
_tf_keras_layer█{"class_name": "Dense", "name": "output_class_distribution", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "output_class_distribution", "trainable": true, "dtype": "float32", "units": 200, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 2048}}}}
Ь
	еiter
жbeta_1
зbeta_2

иdecay
йlearning_rate mЪ!mЫ'mЬ(mЭ;mЮ<mЯBmаCmбRmвSmгYmдZmеimжjmзpmиqmй	Аmк	Бmл	Зmм	Иmн	Яmо	аmп v░!v▒'v▓(v│;v┤<v╡Bv╢Cv╖Rv╕Sv╣Yv║Zv╗iv╝jv╜pv╛qv┐	Аv└	Бv┴	Зv┬	Иv├	Яv─	аv┼"
	optimizer
Ю
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
А24
Б25
З26
И27
Й28
К29
Я30
а31"
trackable_list_wrapper
 "
trackable_list_wrapper
╠
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
А16
Б17
З18
И19
Я20
а21"
trackable_list_wrapper
┐
кlayers
 лlayer_regularization_losses
мmetrics
нnon_trainable_variables
	variables
regularization_losses
trainable_variables
╟__call__
╚_default_save_signature
+╞&call_and_return_all_conditional_losses
'╞"call_and_return_conditional_losses"
_generic_user_object
-
∙serving_default"
signature_map
.:, 2input_conv_14/kernel
 : 2input_conv_14/bias
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
б
оlayers
 пlayer_regularization_losses
░metrics
▒non_trainable_variables
"	variables
#regularization_losses
$trainable_variables
╩__call__
+╔&call_and_return_all_conditional_losses
'╔"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
0:. 2"input_batch_normalization_14/gamma
/:- 2!input_batch_normalization_14/beta
8:6  (2(input_batch_normalization_14/moving_mean
<::  (2,input_batch_normalization_14/moving_variance
<
'0
(1
)2
*3"
trackable_list_wrapper
 "
trackable_list_wrapper
.
'0
(1"
trackable_list_wrapper
б
▓layers
 │layer_regularization_losses
┤metrics
╡non_trainable_variables
+	variables
,regularization_losses
-trainable_variables
╠__call__
+╦&call_and_return_all_conditional_losses
'╦"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
б
╢layers
 ╖layer_regularization_losses
╕metrics
╣non_trainable_variables
/	variables
0regularization_losses
1trainable_variables
╬__call__
+═&call_and_return_all_conditional_losses
'═"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
б
║layers
 ╗layer_regularization_losses
╝metrics
╜non_trainable_variables
3	variables
4regularization_losses
5trainable_variables
╨__call__
+╧&call_and_return_all_conditional_losses
'╧"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
б
╛layers
 ┐layer_regularization_losses
└metrics
┴non_trainable_variables
7	variables
8regularization_losses
9trainable_variables
╥__call__
+╤&call_and_return_all_conditional_losses
'╤"call_and_return_conditional_losses"
_generic_user_object
*:(  2conv_1_14/kernel
: 2conv_1_14/bias
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
б
┬layers
 ├layer_regularization_losses
─metrics
┼non_trainable_variables
=	variables
>regularization_losses
?trainable_variables
╘__call__
+╙&call_and_return_all_conditional_losses
'╙"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
: 2b_norm_1_14/gamma
: 2b_norm_1_14/beta
':%  (2b_norm_1_14/moving_mean
+:)  (2b_norm_1_14/moving_variance
<
B0
C1
D2
E3"
trackable_list_wrapper
 "
trackable_list_wrapper
.
B0
C1"
trackable_list_wrapper
б
╞layers
 ╟layer_regularization_losses
╚metrics
╔non_trainable_variables
F	variables
Gregularization_losses
Htrainable_variables
╓__call__
+╒&call_and_return_all_conditional_losses
'╒"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
б
╩layers
 ╦layer_regularization_losses
╠metrics
═non_trainable_variables
J	variables
Kregularization_losses
Ltrainable_variables
╪__call__
+╫&call_and_return_all_conditional_losses
'╫"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
б
╬layers
 ╧layer_regularization_losses
╨metrics
╤non_trainable_variables
N	variables
Oregularization_losses
Ptrainable_variables
┌__call__
+┘&call_and_return_all_conditional_losses
'┘"call_and_return_conditional_losses"
_generic_user_object
*:( @2conv_2_14/kernel
:@2conv_2_14/bias
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
б
╥layers
 ╙layer_regularization_losses
╘metrics
╒non_trainable_variables
T	variables
Uregularization_losses
Vtrainable_variables
▄__call__
+█&call_and_return_all_conditional_losses
'█"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
:@2b_norm_2_14/gamma
:@2b_norm_2_14/beta
':%@ (2b_norm_2_14/moving_mean
+:)@ (2b_norm_2_14/moving_variance
<
Y0
Z1
[2
\3"
trackable_list_wrapper
 "
trackable_list_wrapper
.
Y0
Z1"
trackable_list_wrapper
б
╓layers
 ╫layer_regularization_losses
╪metrics
┘non_trainable_variables
]	variables
^regularization_losses
_trainable_variables
▐__call__
+▌&call_and_return_all_conditional_losses
'▌"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
б
┌layers
 █layer_regularization_losses
▄metrics
▌non_trainable_variables
a	variables
bregularization_losses
ctrainable_variables
р__call__
+▀&call_and_return_all_conditional_losses
'▀"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
б
▐layers
 ▀layer_regularization_losses
рmetrics
сnon_trainable_variables
e	variables
fregularization_losses
gtrainable_variables
т__call__
+с&call_and_return_all_conditional_losses
'с"call_and_return_conditional_losses"
_generic_user_object
*:(@А2conv_3_9/kernel
:А2conv_3_9/bias
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
б
тlayers
 уlayer_regularization_losses
фmetrics
хnon_trainable_variables
k	variables
lregularization_losses
mtrainable_variables
ф__call__
+у&call_and_return_all_conditional_losses
'у"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
:А2b_norm_3_9/gamma
:А2b_norm_3_9/beta
':%А (2b_norm_3_9/moving_mean
+:)А (2b_norm_3_9/moving_variance
<
p0
q1
r2
s3"
trackable_list_wrapper
 "
trackable_list_wrapper
.
p0
q1"
trackable_list_wrapper
б
цlayers
 чlayer_regularization_losses
шmetrics
щnon_trainable_variables
t	variables
uregularization_losses
vtrainable_variables
ц__call__
+х&call_and_return_all_conditional_losses
'х"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
б
ъlayers
 ыlayer_regularization_losses
ьmetrics
эnon_trainable_variables
x	variables
yregularization_losses
ztrainable_variables
ш__call__
+ч&call_and_return_all_conditional_losses
'ч"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
б
юlayers
 яlayer_regularization_losses
Ёmetrics
ёnon_trainable_variables
|	variables
}regularization_losses
~trainable_variables
ъ__call__
+щ&call_and_return_all_conditional_losses
'щ"call_and_return_conditional_losses"
_generic_user_object
+:)АА2conv_4_4/kernel
:А2conv_4_4/bias
0
А0
Б1"
trackable_list_wrapper
 "
trackable_list_wrapper
0
А0
Б1"
trackable_list_wrapper
д
Єlayers
 єlayer_regularization_losses
Їmetrics
їnon_trainable_variables
В	variables
Гregularization_losses
Дtrainable_variables
ь__call__
+ы&call_and_return_all_conditional_losses
'ы"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
:А2b_norm_4_4/gamma
:А2b_norm_4_4/beta
':%А (2b_norm_4_4/moving_mean
+:)А (2b_norm_4_4/moving_variance
@
З0
И1
Й2
К3"
trackable_list_wrapper
 "
trackable_list_wrapper
0
З0
И1"
trackable_list_wrapper
д
Ўlayers
 ўlayer_regularization_losses
°metrics
∙non_trainable_variables
Л	variables
Мregularization_losses
Нtrainable_variables
ю__call__
+э&call_and_return_all_conditional_losses
'э"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
д
·layers
 √layer_regularization_losses
№metrics
¤non_trainable_variables
П	variables
Рregularization_losses
Сtrainable_variables
Ё__call__
+я&call_and_return_all_conditional_losses
'я"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
д
■layers
  layer_regularization_losses
Аmetrics
Бnon_trainable_variables
У	variables
Фregularization_losses
Хtrainable_variables
Є__call__
+ё&call_and_return_all_conditional_losses
'ё"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
д
Вlayers
 Гlayer_regularization_losses
Дmetrics
Еnon_trainable_variables
Ч	variables
Шregularization_losses
Щtrainable_variables
Ї__call__
+є&call_and_return_all_conditional_losses
'є"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
д
Жlayers
 Зlayer_regularization_losses
Иmetrics
Йnon_trainable_variables
Ы	variables
Ьregularization_losses
Эtrainable_variables
Ў__call__
+ї&call_and_return_all_conditional_losses
'ї"call_and_return_conditional_losses"
_generic_user_object
7:5
А╚2#output_class_distribution_14/kernel
0:.╚2!output_class_distribution_14/bias
0
Я0
а1"
trackable_list_wrapper
 "
trackable_list_wrapper
0
Я0
а1"
trackable_list_wrapper
д
Кlayers
 Лlayer_regularization_losses
Мmetrics
Нnon_trainable_variables
б	variables
вregularization_losses
гtrainable_variables
°__call__
+ў&call_and_return_all_conditional_losses
'ў"call_and_return_conditional_losses"
_generic_user_object
:	 (2	Adam/iter
: (2Adam/beta_1
: (2Adam/beta_2
: (2
Adam/decay
: (2Adam/learning_rate
╓
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
 "
trackable_list_wrapper
(
О0"
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
Й8
К9"
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
Й0
К1"
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
╜

Пtotal

Рcount
С
_fn_kwargs
Т	variables
Уregularization_losses
Фtrainable_variables
Х	keras_api
+·&call_and_return_all_conditional_losses
√__call__" 
_tf_keras_layerх{"class_name": "CategoricalAccuracy", "name": "categorical_accuracy", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "config": {"name": "categorical_accuracy", "dtype": "float32"}}
:  (2total
:  (2count
 "
trackable_dict_wrapper
0
П0
Р1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
д
Цlayers
 Чlayer_regularization_losses
Шmetrics
Щnon_trainable_variables
Т	variables
Уregularization_losses
Фtrainable_variables
√__call__
+·&call_and_return_all_conditional_losses
'·"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
0
П0
Р1"
trackable_list_wrapper
3:1 2Adam/input_conv_14/kernel/m
%:# 2Adam/input_conv_14/bias/m
5:3 2)Adam/input_batch_normalization_14/gamma/m
4:2 2(Adam/input_batch_normalization_14/beta/m
/:-  2Adam/conv_1_14/kernel/m
!: 2Adam/conv_1_14/bias/m
$:" 2Adam/b_norm_1_14/gamma/m
#:! 2Adam/b_norm_1_14/beta/m
/:- @2Adam/conv_2_14/kernel/m
!:@2Adam/conv_2_14/bias/m
$:"@2Adam/b_norm_2_14/gamma/m
#:!@2Adam/b_norm_2_14/beta/m
/:-@А2Adam/conv_3_9/kernel/m
!:А2Adam/conv_3_9/bias/m
$:"А2Adam/b_norm_3_9/gamma/m
#:!А2Adam/b_norm_3_9/beta/m
0:.АА2Adam/conv_4_4/kernel/m
!:А2Adam/conv_4_4/bias/m
$:"А2Adam/b_norm_4_4/gamma/m
#:!А2Adam/b_norm_4_4/beta/m
<::
А╚2*Adam/output_class_distribution_14/kernel/m
5:3╚2(Adam/output_class_distribution_14/bias/m
3:1 2Adam/input_conv_14/kernel/v
%:# 2Adam/input_conv_14/bias/v
5:3 2)Adam/input_batch_normalization_14/gamma/v
4:2 2(Adam/input_batch_normalization_14/beta/v
/:-  2Adam/conv_1_14/kernel/v
!: 2Adam/conv_1_14/bias/v
$:" 2Adam/b_norm_1_14/gamma/v
#:! 2Adam/b_norm_1_14/beta/v
/:- @2Adam/conv_2_14/kernel/v
!:@2Adam/conv_2_14/bias/v
$:"@2Adam/b_norm_2_14/gamma/v
#:!@2Adam/b_norm_2_14/beta/v
/:-@А2Adam/conv_3_9/kernel/v
!:А2Adam/conv_3_9/bias/v
$:"А2Adam/b_norm_3_9/gamma/v
#:!А2Adam/b_norm_3_9/beta/v
0:.АА2Adam/conv_4_4/kernel/v
!:А2Adam/conv_4_4/bias/v
$:"А2Adam/b_norm_4_4/gamma/v
#:!А2Adam/b_norm_4_4/beta/v
<::
А╚2*Adam/output_class_distribution_14/kernel/v
5:3╚2(Adam/output_class_distribution_14/bias/v
ъ2ч
G__inference_awsome_net_layer_call_and_return_conditional_losses_2737310
G__inference_awsome_net_layer_call_and_return_conditional_losses_2736775
G__inference_awsome_net_layer_call_and_return_conditional_losses_2736715
G__inference_awsome_net_layer_call_and_return_conditional_losses_2737454└
╖▓│
FullArgSpec1
args)Ъ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЪ
p 

 

kwonlyargsЪ 
kwonlydefaultsк 
annotationsк *
 
■2√
,__inference_awsome_net_layer_call_fn_2736970
,__inference_awsome_net_layer_call_fn_2737491
,__inference_awsome_net_layer_call_fn_2736873
,__inference_awsome_net_layer_call_fn_2737528└
╖▓│
FullArgSpec1
args)Ъ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЪ
p 

 

kwonlyargsЪ 
kwonlydefaultsк 
annotationsк *
 
є2Ё
"__inference__wrapped_model_2735298╔
Л▓З
FullArgSpec
argsЪ 
varargsjargs
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *9в6
4К1
input_conv_input         АА
ж2г
G__inference_input_conv_layer_call_and_return_conditional_losses_2735310╫
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *7в4
2К/+                           
Л2И
,__inference_input_conv_layer_call_fn_2735318╫
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *7в4
2К/+                           
Ъ2Ч
V__inference_input_batch_normalization_layer_call_and_return_conditional_losses_2737596
V__inference_input_batch_normalization_layer_call_and_return_conditional_losses_2737670
V__inference_input_batch_normalization_layer_call_and_return_conditional_losses_2737648
V__inference_input_batch_normalization_layer_call_and_return_conditional_losses_2737574┤
л▓з
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaultsк 
annotationsк *
 
о2л
;__inference_input_batch_normalization_layer_call_fn_2737614
;__inference_input_batch_normalization_layer_call_fn_2737679
;__inference_input_batch_normalization_layer_call_fn_2737605
;__inference_input_batch_normalization_layer_call_fn_2737688┤
л▓з
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaultsк 
annotationsк *
 
ё2ю
G__inference_input_RELU_layer_call_and_return_conditional_losses_2737693в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
╓2╙
,__inference_input_RELU_layer_call_fn_2737698в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
╡2▓
M__inference_max_pooling2d_60_layer_call_and_return_conditional_losses_2735456р
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *@в=
;К84                                    
Ъ2Ч
2__inference_max_pooling2d_60_layer_call_fn_2735462р
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *@в=
;К84                                    
╠2╔
G__inference_dropout_28_layer_call_and_return_conditional_losses_2737723
G__inference_dropout_28_layer_call_and_return_conditional_losses_2737718┤
л▓з
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaultsк 
annotationsк *
 
Ц2У
,__inference_dropout_28_layer_call_fn_2737728
,__inference_dropout_28_layer_call_fn_2737733┤
л▓з
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaultsк 
annotationsк *
 
в2Я
C__inference_conv_1_layer_call_and_return_conditional_losses_2735474╫
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *7в4
2К/+                            
З2Д
(__inference_conv_1_layer_call_fn_2735482╫
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *7в4
2К/+                            
╓2╙
E__inference_b_norm_1_layer_call_and_return_conditional_losses_2737801
E__inference_b_norm_1_layer_call_and_return_conditional_losses_2737853
E__inference_b_norm_1_layer_call_and_return_conditional_losses_2737875
E__inference_b_norm_1_layer_call_and_return_conditional_losses_2737779┤
л▓з
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaultsк 
annotationsк *
 
ъ2ч
*__inference_b_norm_1_layer_call_fn_2737884
*__inference_b_norm_1_layer_call_fn_2737893
*__inference_b_norm_1_layer_call_fn_2737810
*__inference_b_norm_1_layer_call_fn_2737819┤
л▓з
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaultsк 
annotationsк *
 
э2ъ
C__inference_RELU_1_layer_call_and_return_conditional_losses_2737898в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
╥2╧
(__inference_RELU_1_layer_call_fn_2737903в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
╡2▓
M__inference_max_pooling2d_56_layer_call_and_return_conditional_losses_2735620р
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *@в=
;К84                                    
Ъ2Ч
2__inference_max_pooling2d_56_layer_call_fn_2735626р
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *@в=
;К84                                    
в2Я
C__inference_conv_2_layer_call_and_return_conditional_losses_2735638╫
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *7в4
2К/+                            
З2Д
(__inference_conv_2_layer_call_fn_2735646╫
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *7в4
2К/+                            
╓2╙
E__inference_b_norm_2_layer_call_and_return_conditional_losses_2738045
E__inference_b_norm_2_layer_call_and_return_conditional_losses_2737971
E__inference_b_norm_2_layer_call_and_return_conditional_losses_2738023
E__inference_b_norm_2_layer_call_and_return_conditional_losses_2737949┤
л▓з
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaultsк 
annotationsк *
 
ъ2ч
*__inference_b_norm_2_layer_call_fn_2738063
*__inference_b_norm_2_layer_call_fn_2737980
*__inference_b_norm_2_layer_call_fn_2737989
*__inference_b_norm_2_layer_call_fn_2738054┤
л▓з
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaultsк 
annotationsк *
 
э2ъ
C__inference_RELU_2_layer_call_and_return_conditional_losses_2738068в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
╥2╧
(__inference_RELU_2_layer_call_fn_2738073в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
╡2▓
M__inference_max_pooling2d_57_layer_call_and_return_conditional_losses_2735784р
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *@в=
;К84                                    
Ъ2Ч
2__inference_max_pooling2d_57_layer_call_fn_2735790р
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *@в=
;К84                                    
в2Я
C__inference_conv_3_layer_call_and_return_conditional_losses_2735802╫
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *7в4
2К/+                           @
З2Д
(__inference_conv_3_layer_call_fn_2735810╫
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *7в4
2К/+                           @
╓2╙
E__inference_b_norm_3_layer_call_and_return_conditional_losses_2738119
E__inference_b_norm_3_layer_call_and_return_conditional_losses_2738141
E__inference_b_norm_3_layer_call_and_return_conditional_losses_2738215
E__inference_b_norm_3_layer_call_and_return_conditional_losses_2738193┤
л▓з
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaultsк 
annotationsк *
 
ъ2ч
*__inference_b_norm_3_layer_call_fn_2738159
*__inference_b_norm_3_layer_call_fn_2738233
*__inference_b_norm_3_layer_call_fn_2738224
*__inference_b_norm_3_layer_call_fn_2738150┤
л▓з
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaultsк 
annotationsк *
 
э2ъ
C__inference_RELU_3_layer_call_and_return_conditional_losses_2738238в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
╥2╧
(__inference_RELU_3_layer_call_fn_2738243в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
╡2▓
M__inference_max_pooling2d_58_layer_call_and_return_conditional_losses_2735948р
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *@в=
;К84                                    
Ъ2Ч
2__inference_max_pooling2d_58_layer_call_fn_2735954р
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *@в=
;К84                                    
г2а
C__inference_conv_4_layer_call_and_return_conditional_losses_2735966╪
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *8в5
3К0,                           А
И2Е
(__inference_conv_4_layer_call_fn_2735974╪
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *8в5
3К0,                           А
╓2╙
E__inference_b_norm_4_layer_call_and_return_conditional_losses_2738311
E__inference_b_norm_4_layer_call_and_return_conditional_losses_2738363
E__inference_b_norm_4_layer_call_and_return_conditional_losses_2738385
E__inference_b_norm_4_layer_call_and_return_conditional_losses_2738289┤
л▓з
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaultsк 
annotationsк *
 
ъ2ч
*__inference_b_norm_4_layer_call_fn_2738320
*__inference_b_norm_4_layer_call_fn_2738403
*__inference_b_norm_4_layer_call_fn_2738329
*__inference_b_norm_4_layer_call_fn_2738394┤
л▓з
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaultsк 
annotationsк *
 
э2ъ
C__inference_RELU_4_layer_call_and_return_conditional_losses_2738408в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
╥2╧
(__inference_RELU_4_layer_call_fn_2738413в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
╡2▓
M__inference_max_pooling2d_59_layer_call_and_return_conditional_losses_2736112р
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *@в=
;К84                                    
Ъ2Ч
2__inference_max_pooling2d_59_layer_call_fn_2736118р
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *@в=
;К84                                    
ё2ю
G__inference_flatten_14_layer_call_and_return_conditional_losses_2738419в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
╓2╙
,__inference_flatten_14_layer_call_fn_2738424в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
╠2╔
G__inference_dropout_29_layer_call_and_return_conditional_losses_2738444
G__inference_dropout_29_layer_call_and_return_conditional_losses_2738449┤
л▓з
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaultsк 
annotationsк *
 
Ц2У
,__inference_dropout_29_layer_call_fn_2738459
,__inference_dropout_29_layer_call_fn_2738454┤
л▓з
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaultsк 
annotationsк *
 
А2¤
V__inference_output_class_distribution_layer_call_and_return_conditional_losses_2738469в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
х2т
;__inference_output_class_distribution_layer_call_fn_2738476в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
=B;
%__inference_signature_wrapper_2737076input_conv_input
╠2╔╞
╜▓╣
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkwjkwargs
defaultsЪ 

kwonlyargsЪ

jtraining%
kwonlydefaultsк

trainingp 
annotationsк *
 
╠2╔╞
╜▓╣
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkwjkwargs
defaultsЪ 

kwonlyargsЪ

jtraining%
kwonlydefaultsк

trainingp 
annotationsк *
 п
C__inference_RELU_1_layer_call_and_return_conditional_losses_2737898h7в4
-в*
(К%
inputs         @@ 
к "-в*
#К 
0         @@ 
Ъ З
(__inference_RELU_1_layer_call_fn_2737903[7в4
-в*
(К%
inputs         @@ 
к " К         @@ п
C__inference_RELU_2_layer_call_and_return_conditional_losses_2738068h7в4
-в*
(К%
inputs           @
к "-в*
#К 
0           @
Ъ З
(__inference_RELU_2_layer_call_fn_2738073[7в4
-в*
(К%
inputs           @
к " К           @▒
C__inference_RELU_3_layer_call_and_return_conditional_losses_2738238j8в5
.в+
)К&
inputs         А
к ".в+
$К!
0         А
Ъ Й
(__inference_RELU_3_layer_call_fn_2738243]8в5
.в+
)К&
inputs         А
к "!К         А▒
C__inference_RELU_4_layer_call_and_return_conditional_losses_2738408j8в5
.в+
)К&
inputs         А
к ".в+
$К!
0         А
Ъ Й
(__inference_RELU_4_layer_call_fn_2738413]8в5
.в+
)К&
inputs         А
к "!К         Аю
"__inference__wrapped_model_2735298╟( !'()*;<BCDERSYZ[\ijpqrsАБЗИЙКЯаCв@
9в6
4К1
input_conv_input         АА
к "VкS
Q
output_class_distribution4К1
output_class_distribution         ╚ы
G__inference_awsome_net_layer_call_and_return_conditional_losses_2736715Я( !'()*;<BCDERSYZ[\ijpqrsАБЗИЙКЯаKвH
Aв>
4К1
input_conv_input         АА
p

 
к "&в#
К
0         ╚
Ъ ы
G__inference_awsome_net_layer_call_and_return_conditional_losses_2736775Я( !'()*;<BCDERSYZ[\ijpqrsАБЗИЙКЯаKвH
Aв>
4К1
input_conv_input         АА
p 

 
к "&в#
К
0         ╚
Ъ с
G__inference_awsome_net_layer_call_and_return_conditional_losses_2737310Х( !'()*;<BCDERSYZ[\ijpqrsАБЗИЙКЯаAв>
7в4
*К'
inputs         АА
p

 
к "&в#
К
0         ╚
Ъ с
G__inference_awsome_net_layer_call_and_return_conditional_losses_2737454Х( !'()*;<BCDERSYZ[\ijpqrsАБЗИЙКЯаAв>
7в4
*К'
inputs         АА
p 

 
к "&в#
К
0         ╚
Ъ ├
,__inference_awsome_net_layer_call_fn_2736873Т( !'()*;<BCDERSYZ[\ijpqrsАБЗИЙКЯаKвH
Aв>
4К1
input_conv_input         АА
p

 
к "К         ╚├
,__inference_awsome_net_layer_call_fn_2736970Т( !'()*;<BCDERSYZ[\ijpqrsАБЗИЙКЯаKвH
Aв>
4К1
input_conv_input         АА
p 

 
к "К         ╚╣
,__inference_awsome_net_layer_call_fn_2737491И( !'()*;<BCDERSYZ[\ijpqrsАБЗИЙКЯаAв>
7в4
*К'
inputs         АА
p

 
к "К         ╚╣
,__inference_awsome_net_layer_call_fn_2737528И( !'()*;<BCDERSYZ[\ijpqrsАБЗИЙКЯаAв>
7в4
*К'
inputs         АА
p 

 
к "К         ╚╗
E__inference_b_norm_1_layer_call_and_return_conditional_losses_2737779rBCDE;в8
1в.
(К%
inputs         @@ 
p
к "-в*
#К 
0         @@ 
Ъ ╗
E__inference_b_norm_1_layer_call_and_return_conditional_losses_2737801rBCDE;в8
1в.
(К%
inputs         @@ 
p 
к "-в*
#К 
0         @@ 
Ъ р
E__inference_b_norm_1_layer_call_and_return_conditional_losses_2737853ЦBCDEMвJ
Cв@
:К7
inputs+                            
p
к "?в<
5К2
0+                            
Ъ р
E__inference_b_norm_1_layer_call_and_return_conditional_losses_2737875ЦBCDEMвJ
Cв@
:К7
inputs+                            
p 
к "?в<
5К2
0+                            
Ъ У
*__inference_b_norm_1_layer_call_fn_2737810eBCDE;в8
1в.
(К%
inputs         @@ 
p
к " К         @@ У
*__inference_b_norm_1_layer_call_fn_2737819eBCDE;в8
1в.
(К%
inputs         @@ 
p 
к " К         @@ ╕
*__inference_b_norm_1_layer_call_fn_2737884ЙBCDEMвJ
Cв@
:К7
inputs+                            
p
к "2К/+                            ╕
*__inference_b_norm_1_layer_call_fn_2737893ЙBCDEMвJ
Cв@
:К7
inputs+                            
p 
к "2К/+                            р
E__inference_b_norm_2_layer_call_and_return_conditional_losses_2737949ЦYZ[\MвJ
Cв@
:К7
inputs+                           @
p
к "?в<
5К2
0+                           @
Ъ р
E__inference_b_norm_2_layer_call_and_return_conditional_losses_2737971ЦYZ[\MвJ
Cв@
:К7
inputs+                           @
p 
к "?в<
5К2
0+                           @
Ъ ╗
E__inference_b_norm_2_layer_call_and_return_conditional_losses_2738023rYZ[\;в8
1в.
(К%
inputs           @
p
к "-в*
#К 
0           @
Ъ ╗
E__inference_b_norm_2_layer_call_and_return_conditional_losses_2738045rYZ[\;в8
1в.
(К%
inputs           @
p 
к "-в*
#К 
0           @
Ъ ╕
*__inference_b_norm_2_layer_call_fn_2737980ЙYZ[\MвJ
Cв@
:К7
inputs+                           @
p
к "2К/+                           @╕
*__inference_b_norm_2_layer_call_fn_2737989ЙYZ[\MвJ
Cв@
:К7
inputs+                           @
p 
к "2К/+                           @У
*__inference_b_norm_2_layer_call_fn_2738054eYZ[\;в8
1в.
(К%
inputs           @
p
к " К           @У
*__inference_b_norm_2_layer_call_fn_2738063eYZ[\;в8
1в.
(К%
inputs           @
p 
к " К           @т
E__inference_b_norm_3_layer_call_and_return_conditional_losses_2738119ШpqrsNвK
DвA
;К8
inputs,                           А
p
к "@в=
6К3
0,                           А
Ъ т
E__inference_b_norm_3_layer_call_and_return_conditional_losses_2738141ШpqrsNвK
DвA
;К8
inputs,                           А
p 
к "@в=
6К3
0,                           А
Ъ ╜
E__inference_b_norm_3_layer_call_and_return_conditional_losses_2738193tpqrs<в9
2в/
)К&
inputs         А
p
к ".в+
$К!
0         А
Ъ ╜
E__inference_b_norm_3_layer_call_and_return_conditional_losses_2738215tpqrs<в9
2в/
)К&
inputs         А
p 
к ".в+
$К!
0         А
Ъ ║
*__inference_b_norm_3_layer_call_fn_2738150ЛpqrsNвK
DвA
;К8
inputs,                           А
p
к "3К0,                           А║
*__inference_b_norm_3_layer_call_fn_2738159ЛpqrsNвK
DвA
;К8
inputs,                           А
p 
к "3К0,                           АХ
*__inference_b_norm_3_layer_call_fn_2738224gpqrs<в9
2в/
)К&
inputs         А
p
к "!К         АХ
*__inference_b_norm_3_layer_call_fn_2738233gpqrs<в9
2в/
)К&
inputs         А
p 
к "!К         А┴
E__inference_b_norm_4_layer_call_and_return_conditional_losses_2738289xЗИЙК<в9
2в/
)К&
inputs         А
p
к ".в+
$К!
0         А
Ъ ┴
E__inference_b_norm_4_layer_call_and_return_conditional_losses_2738311xЗИЙК<в9
2в/
)К&
inputs         А
p 
к ".в+
$К!
0         А
Ъ ц
E__inference_b_norm_4_layer_call_and_return_conditional_losses_2738363ЬЗИЙКNвK
DвA
;К8
inputs,                           А
p
к "@в=
6К3
0,                           А
Ъ ц
E__inference_b_norm_4_layer_call_and_return_conditional_losses_2738385ЬЗИЙКNвK
DвA
;К8
inputs,                           А
p 
к "@в=
6К3
0,                           А
Ъ Щ
*__inference_b_norm_4_layer_call_fn_2738320kЗИЙК<в9
2в/
)К&
inputs         А
p
к "!К         АЩ
*__inference_b_norm_4_layer_call_fn_2738329kЗИЙК<в9
2в/
)К&
inputs         А
p 
к "!К         А╛
*__inference_b_norm_4_layer_call_fn_2738394ПЗИЙКNвK
DвA
;К8
inputs,                           А
p
к "3К0,                           А╛
*__inference_b_norm_4_layer_call_fn_2738403ПЗИЙКNвK
DвA
;К8
inputs,                           А
p 
к "3К0,                           А╪
C__inference_conv_1_layer_call_and_return_conditional_losses_2735474Р;<IвF
?в<
:К7
inputs+                            
к "?в<
5К2
0+                            
Ъ ░
(__inference_conv_1_layer_call_fn_2735482Г;<IвF
?в<
:К7
inputs+                            
к "2К/+                            ╪
C__inference_conv_2_layer_call_and_return_conditional_losses_2735638РRSIвF
?в<
:К7
inputs+                            
к "?в<
5К2
0+                           @
Ъ ░
(__inference_conv_2_layer_call_fn_2735646ГRSIвF
?в<
:К7
inputs+                            
к "2К/+                           @┘
C__inference_conv_3_layer_call_and_return_conditional_losses_2735802СijIвF
?в<
:К7
inputs+                           @
к "@в=
6К3
0,                           А
Ъ ▒
(__inference_conv_3_layer_call_fn_2735810ДijIвF
?в<
:К7
inputs+                           @
к "3К0,                           А▄
C__inference_conv_4_layer_call_and_return_conditional_losses_2735966ФАБJвG
@в=
;К8
inputs,                           А
к "@в=
6К3
0,                           А
Ъ ┤
(__inference_conv_4_layer_call_fn_2735974ЗАБJвG
@в=
;К8
inputs,                           А
к "3К0,                           А╖
G__inference_dropout_28_layer_call_and_return_conditional_losses_2737718l;в8
1в.
(К%
inputs         @@ 
p
к "-в*
#К 
0         @@ 
Ъ ╖
G__inference_dropout_28_layer_call_and_return_conditional_losses_2737723l;в8
1в.
(К%
inputs         @@ 
p 
к "-в*
#К 
0         @@ 
Ъ П
,__inference_dropout_28_layer_call_fn_2737728_;в8
1в.
(К%
inputs         @@ 
p
к " К         @@ П
,__inference_dropout_28_layer_call_fn_2737733_;в8
1в.
(К%
inputs         @@ 
p 
к " К         @@ й
G__inference_dropout_29_layer_call_and_return_conditional_losses_2738444^4в1
*в'
!К
inputs         А
p
к "&в#
К
0         А
Ъ й
G__inference_dropout_29_layer_call_and_return_conditional_losses_2738449^4в1
*в'
!К
inputs         А
p 
к "&в#
К
0         А
Ъ Б
,__inference_dropout_29_layer_call_fn_2738454Q4в1
*в'
!К
inputs         А
p
к "К         АБ
,__inference_dropout_29_layer_call_fn_2738459Q4в1
*в'
!К
inputs         А
p 
к "К         Ан
G__inference_flatten_14_layer_call_and_return_conditional_losses_2738419b8в5
.в+
)К&
inputs         А
к "&в#
К
0         А
Ъ Е
,__inference_flatten_14_layer_call_fn_2738424U8в5
.в+
)К&
inputs         А
к "К         А╖
G__inference_input_RELU_layer_call_and_return_conditional_losses_2737693l9в6
/в,
*К'
inputs         АА 
к "/в,
%К"
0         АА 
Ъ П
,__inference_input_RELU_layer_call_fn_2737698_9в6
/в,
*К'
inputs         АА 
к ""К         АА ╨
V__inference_input_batch_normalization_layer_call_and_return_conditional_losses_2737574v'()*=в:
3в0
*К'
inputs         АА 
p
к "/в,
%К"
0         АА 
Ъ ╨
V__inference_input_batch_normalization_layer_call_and_return_conditional_losses_2737596v'()*=в:
3в0
*К'
inputs         АА 
p 
к "/в,
%К"
0         АА 
Ъ ё
V__inference_input_batch_normalization_layer_call_and_return_conditional_losses_2737648Ц'()*MвJ
Cв@
:К7
inputs+                            
p
к "?в<
5К2
0+                            
Ъ ё
V__inference_input_batch_normalization_layer_call_and_return_conditional_losses_2737670Ц'()*MвJ
Cв@
:К7
inputs+                            
p 
к "?в<
5К2
0+                            
Ъ и
;__inference_input_batch_normalization_layer_call_fn_2737605i'()*=в:
3в0
*К'
inputs         АА 
p
к ""К         АА и
;__inference_input_batch_normalization_layer_call_fn_2737614i'()*=в:
3в0
*К'
inputs         АА 
p 
к ""К         АА ╔
;__inference_input_batch_normalization_layer_call_fn_2737679Й'()*MвJ
Cв@
:К7
inputs+                            
p
к "2К/+                            ╔
;__inference_input_batch_normalization_layer_call_fn_2737688Й'()*MвJ
Cв@
:К7
inputs+                            
p 
к "2К/+                            ▄
G__inference_input_conv_layer_call_and_return_conditional_losses_2735310Р !IвF
?в<
:К7
inputs+                           
к "?в<
5К2
0+                            
Ъ ┤
,__inference_input_conv_layer_call_fn_2735318Г !IвF
?в<
:К7
inputs+                           
к "2К/+                            Ё
M__inference_max_pooling2d_56_layer_call_and_return_conditional_losses_2735620ЮRвO
HвE
CК@
inputs4                                    
к "HвE
>К;
04                                    
Ъ ╚
2__inference_max_pooling2d_56_layer_call_fn_2735626СRвO
HвE
CК@
inputs4                                    
к ";К84                                    Ё
M__inference_max_pooling2d_57_layer_call_and_return_conditional_losses_2735784ЮRвO
HвE
CК@
inputs4                                    
к "HвE
>К;
04                                    
Ъ ╚
2__inference_max_pooling2d_57_layer_call_fn_2735790СRвO
HвE
CК@
inputs4                                    
к ";К84                                    Ё
M__inference_max_pooling2d_58_layer_call_and_return_conditional_losses_2735948ЮRвO
HвE
CК@
inputs4                                    
к "HвE
>К;
04                                    
Ъ ╚
2__inference_max_pooling2d_58_layer_call_fn_2735954СRвO
HвE
CК@
inputs4                                    
к ";К84                                    Ё
M__inference_max_pooling2d_59_layer_call_and_return_conditional_losses_2736112ЮRвO
HвE
CК@
inputs4                                    
к "HвE
>К;
04                                    
Ъ ╚
2__inference_max_pooling2d_59_layer_call_fn_2736118СRвO
HвE
CК@
inputs4                                    
к ";К84                                    Ё
M__inference_max_pooling2d_60_layer_call_and_return_conditional_losses_2735456ЮRвO
HвE
CК@
inputs4                                    
к "HвE
>К;
04                                    
Ъ ╚
2__inference_max_pooling2d_60_layer_call_fn_2735462СRвO
HвE
CК@
inputs4                                    
к ";К84                                    ║
V__inference_output_class_distribution_layer_call_and_return_conditional_losses_2738469`Яа0в-
&в#
!К
inputs         А
к "&в#
К
0         ╚
Ъ Т
;__inference_output_class_distribution_layer_call_fn_2738476SЯа0в-
&в#
!К
inputs         А
к "К         ╚Е
%__inference_signature_wrapper_2737076█( !'()*;<BCDERSYZ[\ijpqrsАБЗИЙКЯаWвT
в 
MкJ
H
input_conv_input4К1
input_conv_input         АА"VкS
Q
output_class_distribution4К1
output_class_distribution         ╚