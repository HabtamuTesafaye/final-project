from rest_framework import serializers
from TimeSeriesBase.models import DashboardTopic , Category

class DashboardTopicSerializer(serializers.ModelSerializer):
    category_count = serializers.IntegerField(read_only=True)
    class Meta:
        model = DashboardTopic
        fields =  '__all__'




class CategorySerializer(serializers.ModelSerializer):
    category_count = serializers.IntegerField(read_only=True)
    class Meta:
        model = Category
        fields =  '__all__'


