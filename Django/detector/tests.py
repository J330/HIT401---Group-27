from django.test import SimpleTestCase
from django.urls import reverse


class DetectorPageTests(SimpleTestCase):
    def test_home_page(self):
        response = self.client.get(reverse('detector:home'))
        self.assertEqual(response.status_code, 200)

    def test_upload_page(self):
        response = self.client.get(reverse('detector:upload'))
        self.assertEqual(response.status_code, 200)

    def test_result_page(self):
        response = self.client.get(reverse('detector:result'))
        self.assertEqual(response.status_code, 200)
