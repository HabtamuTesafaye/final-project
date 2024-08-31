# decorators.py

from functools import wraps
from django.shortcuts import redirect
from .models import SiteConfiguration
from django.http import HttpResponseForbidden
from django.shortcuts import get_object_or_404

def public_required(view_func):
    @wraps(view_func)
    def _wrapped_view(request, *args, **kwargs):
        # Check if the site is not public
        site_config = get_object_or_404(SiteConfiguration, id=1)  # Adjust the lookup based on your model logic
        if site_config and not site_config.is_public:
            # Check if the user is not authenticated
            if not request.user.is_authenticated:
                # Redirect to the login page
                return redirect('login')

        # Call the original view function
        return view_func(request, *args, **kwargs)

    return _wrapped_view
