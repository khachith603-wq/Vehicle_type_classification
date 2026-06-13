from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('classifier', '0001_initial'),
    ]

    operations = [
        migrations.AddField(
            model_name='prediction',
            name='vehicle_type',
            field=models.CharField(default='Unknown', max_length=100),
        ),
    ]
