from django.urls import path
from . import views

urlpatterns = [
    path('', views.home, name="home"),
    path('create_embeddings/', views.create_embeddings, name="create_embeddings"),
    path('question_answer/', views.question_answer, name='question_answer'),
]