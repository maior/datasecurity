from rest_framework import serializers
from .models import Malicious

#class MaliciousSerializer(serializers.ModelSerializer):
class MaliciousSerializer(serializers.Serializer):
	task = serializers.CharField(required=True)