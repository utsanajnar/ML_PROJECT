from django.shortcuts import render, redirect  
from employee.forms import EmployeeForm  
from employee.models import Employee  
# Create your views here.  
def emp(request):  
    if request.method == "POST":  
        form = EmployeeForm(request.POST)  
        if form.is_valid():  
            try:  
                form.save()  
                return redirect('/show')  
            except:  
                pass  
    else:  
        form = EmployeeForm()  
    return render(request,'index.html',{'form':form})   # render(request , template name , context name)
def show(request):  
    employees = Employee.objects.all()  
    return render(request,"show.html",{'employees':employees})  
def edit(request, id):  
    employee = Employee.objects.get(id=id)  
    return render(request,'edit.html', {'employee':employee})  
def update(request, id):  
    employee = Employee.objects.get(id=id)  
    form = EmployeeForm(request.POST, instance = employee)  
    if form.is_valid():  
        form.save()  
        return redirect("/show")  
    return render(request, 'edit.html', {'employee': employee})  
def destroy(request, id):  
    employee = Employee.objects.get(id=id)  
    employee.delete()  
    return redirect("/show")  


def home(request):
   # return HttpResponse("hello  !!!!!!")
  return render(request ,'home.html')

def about(request):
  #  return HttpResponse("this is ABOUT info") 
  return render(request ,'about.html')


def accountInfo(request):
   # return HttpResponse("this is CONTACTS info") 
  # if request.method=='POST':
  #   firstName=request.POST['firstName']
  #   lastName=request.POST['lastName']
  #   primaryEmail=request.POST['primaryEmail']
  #   secondaryEmail=request.POST['secondaryEmail']
  #   primaryPhone=request.POST['primaryPhone']
  #   secondaryPhone=request.POST['secondaryPhone']
  #   district=request.POST['district']
  #   state=request.POST['state']
  #   pin=request.POST['pin']
  #   ins = accountInfo(firstName=firstName,lastName=lastName,primaryEmail=primaryEmail,secondaryEmail=secondaryEmail,primaryPhone=primaryPhone,secondaryPhone=secondaryPhone,district=district,state=state,pin=pin)
  #   ins.save()
  #   print("the data has been entered!!!")
    
  return render(request ,'accountInfo.html')

def projects(request):
  return render(request , 'projects.html')


def videoScreen(request):
  context ={
        "range3":[1,2,3],
        "range5":[1,2,3,4,5],
        "range9":[1,2,3,4,5,6,7,8,9]
    }
  return render(request,'video3.html',context)

def videoScreen5(request):
  context ={
        "range3":[1,2,3],
        "range5":[1,2,3,4,5],
        "range9":[1,2,3,4,5,6,7,8,9]
    }
  return render(request,'vid5.html',context)

def videoScreen9(request):
  context ={
        "range3":[1,2,3],
        "range5":[1,2,3,4,5],
        "range9":[1,2,3,4,5,6,7,8,9]
    }
  return render(request,'vid9.html',context) 