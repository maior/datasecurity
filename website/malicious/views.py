#rom django.shortcuts import render
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from rest_framework import permissions
from .models import Malicious
from .serializers import MaliciousSerializer
from malicious import serializers

import logging
logger = logging.getLogger(__name__)

from .services import (
    malicious,                           # malicious
)
# Create your views here.
class MaliciousApiView(APIView):
	def post(self, request):
		params = serializers.MaliciousSerializer(data=request.data)

		if not params.is_valid():
			return Response(params.errors, status=status.HTTP_400_BAD_REQUEST)

		service = malicious.Service()
		res = service.run(params.validated_data)
		return Response(res)