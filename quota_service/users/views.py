from rest_framework.decorators import api_view
from rest_framework.response import Response
from .models import UserQuota
from datetime import date

@api_view(['POST'])
def check_quota(request):
    email = request.data.get('email')
    if not email:
        return Response({"error": "Email required"}, status=400)

    user, _ = UserQuota.objects.get_or_create(email=email)

    # Reset count if it's a new day
    if user.date != date.today():
        user.date = date.today()
        user.question_count = 0
        user.save()

    if user.question_count >= 50:
        return Response({"allowed": False, "message": "Quota exceeded"})

    user.question_count += 1
    user.save()
    return Response({"allowed": True, "remaining": 50 - user.question_count})
