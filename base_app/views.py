from django.shortcuts import render
import time
from django.http import StreamingHttpResponse, JsonResponse
from django.views.decorators.csrf import csrf_exempt
from .download_drive_files import embeddings
import json

from .search_embeddings import (load_faiss_index, load_embeddings, embed_question,
                                search_similar, get_related_text, generate_concise_answer)


# Load FAISS index and embeddings
index = load_faiss_index()
embeddings, filenames = load_embeddings()



def home(request):
    return render(request, 'base_app/chatbot.html')



@csrf_exempt
def create_embeddings(request):
    if request.method == 'POST':
        # call embeddings function to create embeddings
        embeddings()
        return JsonResponse({'status': 'Embeddings created successfully'})

    return JsonResponse({'error': 'Invalid request'}, status=400)



@csrf_exempt
def question_answer(request):
    if request.method == 'POST':
        body_unicode = request.body.decode('utf-8')
        body_data = json.loads(body_unicode)
        
        # Extract the user message
        question = body_data.get('message')
        # Embed the question
        question_embedding = embed_question(question)
        # Search for similar embeddings in FAISS
        distances, indices = search_similar(question_embedding, index, top_k=1)
        # Retrieve related file or text
        related_files = get_related_text(indices, filenames)
        context = ''
        if related_files:
            with open(related_files[0], 'r') as f:
                context = f.read()
        
        concise_answer = generate_concise_answer(question, context)
        
        return JsonResponse({"answer": concise_answer})
    
    return JsonResponse({"error": "Invalid request method"}, status=400)
